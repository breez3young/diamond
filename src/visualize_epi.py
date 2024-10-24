from data.episode import Episode
import matplotlib.pyplot as plt
import cv2
import torch
import ipdb

def vis_episode(epi_path, output_file='./ma_test.mp4'):
    episode = Episode.load(epi_path)
    observations = episode.obs

    if observations.ndim > 4:
        observations = observations.mean(dim=1)

    (h, w) = observations.shape[-2:]
    observations = observations.permute(0, 2, 3, 1).clamp(-1, 1).add(1).div(2).mul(255).byte()
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (w, h))  # 20.0是帧率

    for idx in range(observations.shape[0]):
        cur_img = cv2.cvtColor(observations[idx].cpu().numpy(), cv2.COLOR_RGB2BGR)
        out.write(cur_img)

    out.release()

    print("Video has been saved to", output_file)

# def test()

if __name__ == "__main__":
    # path = "/home/eriri/Projects/diamond/outputs/2024-10-23/19-31-38/dataset/test/000/00/3/3.pt"
    path = "/home/eriri/Projects/diamond/outputs/2024-10-23/15-26-11/dataset/test/000/00/3/3.pt"
    "/home/eriri/Projects/diamond/outputs/2024-10-23/20-06-46/dataset/test/000/00/3/3.pt"

    ipdb.set_trace()
    tmp = torch.load("/home/eriri/Projects/diamond/outputs/2024-10-23/15-26-11/dataset/test/info.pt")

    vis_episode(path)