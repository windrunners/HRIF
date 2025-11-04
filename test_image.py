import torch
from torch.autograd import Variable
from net import TUFusion_net
import utils
from args_fusion import args
import numpy as np
import os
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import argparse
import torchvision.transforms as tfs
import torchvision.utils as vutils
from models import FFA4
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dark_channel(img, patch_size=15):
    """计算暗通道"""
    if isinstance(img, Image.Image):
        img = np.array(img)
    if len(img.shape) == 2:  # Grayscale image
        img = np.stack([img]*3, axis=-1)
    min_channel = np.min(img, axis=2)  # 取RGB三通道最小值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)  # 最小值滤波
    return dark_channel


def detect_smoke_by_dark_channel(img, dark_threshold=30, var_threshold=500):
    """通过暗通道判断烟雾"""
    dark = compute_dark_channel(img)
    mean_dark = np.mean(dark)
    var_dark = np.var(dark)

    # 烟雾图像的暗通道均值较高且方差较低
    if mean_dark > dark_threshold and var_dark < var_threshold:
        return True  # 烟雾图像
    else:
        return False  # 清晰图像


def load_ffa_model():
    """加载FFA4模型"""
    model = FFA4(gps=3, blocks=3)
    model_path = './weights/ffa4_best.pth'

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"FFA4 model not found at {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckp = torch.load(model_path, map_location=device)

    # 处理state_dict中的键名
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in ckp['model'].items():
        if k.startswith('module.'):
            name = k[7:]  # 移除'module.'前缀
        else:
            name = k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model.to(device)


def process_visible_image(ffa_model, image_path, temp_dir='temp_processed'):
    """
    使用FFA4模型处理图像
    :param ffa_model: 加载的FFA4模型
    :param image_path: 原始图像路径
    :param temp_dir: 临时保存处理结果的目录
    :return: 处理后的图像路径
    """
    os.makedirs(temp_dir, exist_ok=True)

    # 获取文件名并创建输出路径
    filename = os.path.basename(image_path)
    output_path = os.path.join(temp_dir, filename)

    # 加载和预处理图像
    haze = Image.open(image_path).convert('RGB')
    haze_tensor = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(haze).unsqueeze(0).to(device)

    # 处理图像
    with torch.no_grad():
        pred = ffa_model(haze_tensor)

    # 保存处理后的图像
    vutils.save_image(pred.clamp(0, 1).cpu(), output_path)

    return output_path


def load_model(path, input_nc, output_nc):
    TUFusion_model = TUFusion_net(input_nc, output_nc)
    TUFusion_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in TUFusion_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(TUFusion_model._get_name(), para * type_size / 1000 / 1000))

    TUFusion_model.eval()
    TUFusion_model.to(device)

    return TUFusion_model


def _generate_fusion_image(model, strategy_type, img1, img2, p_type):
    # encoder
    en_r = model.encoder(img1)
    en_v = model.encoder(img2)

    # fusion: hybrid, channel and spatial
    # f = model.fusion(en_r, en_v, p_type)

    # fusion: addition
    f = model.fusion1(en_r, en_v)

    # fusion: composite attention
    # f = model.fusion2(en_r, en_v, p_type)

    # decoder
    img_fusion = model.decoder(f)
    return img_fusion[0]


def convert_to_color(fusion_path, visible_path):
    """
    Convert grayscale fusion image to color using YCbCr from visible image
    """
    # Open images
    vi_img = Image.open(visible_path).convert('YCbCr')
    f_img = Image.open(fusion_path).convert('L')

    # Split channels
    vi_Y, vi_Cb, vi_Cr = vi_img.split()

    # Merge fusion Y with visible CbCr
    f_img_color = Image.merge('YCbCr', (f_img, vi_Cb, vi_Cr)).convert('RGB')

    # Save back to the same path
    f_img_color.save(fusion_path)


def run_demo(model, ffa_model, infrared_path, visible_path, output_path_root, image, fusion_type, network_type,
             strategy_type, mode, p_type):

    # Load original images for smoke detection
    ir_img_orig = Image.open(infrared_path)
    vis_img_orig = Image.open(visible_path)

    # Detect smoke in both images
    ir_has_smoke = detect_smoke_by_dark_channel(ir_img_orig)
    vis_has_smoke = detect_smoke_by_dark_channel(vis_img_orig)

    # Process images based on smoke detection
    processed_ir_path = infrared_path
    processed_vis_path = visible_path

    if ir_has_smoke:
        processed_ir_path = process_visible_image(ffa_model, infrared_path)
    if vis_has_smoke:
        processed_vis_path = process_visible_image(ffa_model, visible_path)

    # Load images for fusion
    ir_img = utils.get_test_images(processed_ir_path, height=None, width=None, mode='L')
    vis_img = utils.get_test_images(processed_vis_path, height=None, width=None, mode='RGB')

    if args.cuda:
        ir_img = ir_img.to(device)
        vis_img = vis_img.to(device)
    ir_img = Variable(ir_img, requires_grad=False)
    vis_img = Variable(vis_img, requires_grad=False)
    dimension = ir_img.size()

    ir_img_resize = F.interpolate(ir_img, size=(256, 256), mode='bilinear', align_corners=False)
    vis_img_resize = F.interpolate(vis_img, size=(256, 256), mode='bilinear', align_corners=False)

    # Convert visible image to Y channel only for fusion
    vis_img_gray = torch.mean(vis_img_resize, dim=1, keepdim=True)

    img_fusion = _generate_fusion_image(model, strategy_type, ir_img_resize, vis_img_gray, p_type)
    img_fusion = F.interpolate(img_fusion, size=(dimension[2], dimension[3]), mode='bilinear', align_corners=False)

    # Save temporary grayscale fusion image
    file_name = image
    temp_output_path = output_path_root + 'temp_' + file_name
    final_output_path = output_path_root + file_name

    # save temporary grayscale image
    if args.cuda:
        img = img_fusion.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = img_fusion.clamp(0, 255).data[0].numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    utils.save_images(temp_output_path, img)

    # Convert to color using the visible image's color information
    convert_to_color(temp_output_path, processed_vis_path)

    # 解决方案1：先检查并删除已存在的文件
    if os.path.exists(final_output_path):
        os.remove(final_output_path)
    os.rename(temp_output_path, final_output_path)

    # 删除临时处理的图像
    if ir_has_smoke and os.path.exists(processed_ir_path):
        os.remove(processed_ir_path)
    if vis_has_smoke and os.path.exists(processed_vis_path):
        os.remove(processed_vis_path)

    print(final_output_path)


def get_images_in_folder(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_names = []
    for filename in os.listdir(folder_path):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_names.append(filename)
    return image_names


def main():
    test_path1 = "test_images/IR/"
    test_path2 = "test_images/VIS/"
    images = get_images_in_folder(test_path1)
    network_type = 'TUfusion'
    strategy_type_list = ['addition', 'attention_weight']
    output_path = './outputs/'
    strategy_type = strategy_type_list[0]
    fusion_type = ['attention_max']
    p_type = fusion_type[0]

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    # in_c = 3 for RGB images; in_c = 1 for gray images
    in_c = 1  # We'll process IR as grayscale
    out_c = in_c
    mode = 'L'  # Grayscale mode for IR

    model_path = args.model_path_gray

    with torch.no_grad():
        model = load_model(model_path, in_c, out_c)
        ffa_model = load_ffa_model()  # 加载FFA4模型

        for image in tqdm(images, desc="Processing images"):
            print(image)
            infrared_path = test_path1 + image
            visible_path = test_path2 + image
            run_demo(model, ffa_model, infrared_path, visible_path, output_path, image, fusion_type, network_type,
                     strategy_type, mode, p_type)
    print('Done......')


if __name__ == '__main__':
    main()