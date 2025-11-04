import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd  # 用于读取Excel

# --- 配置部分 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos")          # 视频文件存放目录
ANNOTATION_FILE = os.path.join(PROJECT_ROOT, "csl_daily_data.xlsx") # Excel 文件路径

# --- 优化: 增加 MAX_FRAMES 以适应数据集 ---
MAX_FRAMES = 450  # 👈 修改: 从 100 增加到 450，确保覆盖数据集中最长的 gloss 序列
BATCH_SIZE = 2
NUM_EPOCHS = 30  # 👈 修改: 减少训练轮数至30，避免过拟合
LEARNING_RATE = 5e-4  # 👈 修改: 初始学习率，配合OneCycleLR
TARGET_SIZE = (224, 224)
HIDDEN_SIZE = 128  # 👈 修改: 减少隐藏层大小
NUM_LAYERS = 1  # 👈 修改: 减少LSTM层数
DROPOUT = 0.4  # 👈 修改: 增加dropout
NUM_WORKERS = 4  # 根据 CPU 核心数调整
EARLY_STOP_PATIENCE = 8  # 👈 修改: 减少早停耐心值，配合更短的训练周期
MIN_DELTA = 0.01  # 早停最小改进阈值

# --- 1. 解析标注文档并建立映射 ---
def parse_annotation_file(annotation_file_path):
    """
    解析 CSL-Daily 的 Excel 标注文件，并建立 视频标识符 -> 文本 的映射。
    从 Excel 文件的 'name' 列获取 ID，从 'gloss' 列获取文本。
    """
    print(f"Parsing annotation file: {annotation_file_path}")
    df = pd.read_excel(annotation_file_path, engine='openpyxl')

    if 'name' not in df.columns or 'gloss' not in df.columns:
        raise ValueError(f"Excel file must contain 'name' and 'gloss' columns. Found columns: {list(df.columns)}")

    df_clean = df[['name', 'gloss']].dropna(subset=['name', 'gloss'])
    annotation_map = dict(zip(df_clean['name'], df_clean['gloss']))

    print(f"Successfully parsed {len(annotation_map)} unique annotations from Excel.")
    return annotation_map


# --- 2. 构建本地数据列表 (新增: 检查视频帧数) ---
def build_local_data_list(video_dir, annotation_file_path, max_frames=MAX_FRAMES):
    """
    构建本地数据列表。
    Args:
        video_dir (str): 存放所有 .mp4 视频文件的目录。
        annotation_file_path (str): 标注文件路径 (Excel)。
        max_frames (int): 最大允许帧数。
    Returns:
        data_list (list): 包含字典的列表，每个字典包含 'video_path', 'text', 'video_id'。
    """
    print("Parsing annotations from Excel...")
    annotation_map = parse_annotation_file(annotation_file_path)

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    print(f"Found {len(video_files)} video files in '{video_dir}'.")

    data_list = []
    missing_videos = 0
    invalid_videos = 0  # 新增计数器

    for video_id in tqdm(annotation_map.keys(), desc="Building dataset from annotations"):
        text = annotation_map[video_id]

        expected_video_filename = f"{video_id}.mp4"
        expected_video_path = os.path.join(video_dir, expected_video_filename)

        if expected_video_filename in video_files:
            # --- 新增: 检查视频是否有效且帧数合适 ---
            cap = cv2.VideoCapture(expected_video_path)
            if not cap.isOpened():
                print(f"Warning: Failed to open video file '{expected_video_filename}' for ID '{video_id}'. Skipping.")
                invalid_videos += 1
                cap.release()
                continue
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames <= 0:
                print(f"Warning: Video file '{expected_video_filename}' has no frames for ID '{video_id}'. Skipping.")
                invalid_videos += 1
                continue
            
            if total_frames > max_frames:
                print(f"Warning: Video file '{expected_video_filename}' has {total_frames} frames, exceeding max_frames {max_frames} for ID '{video_id}'. Skipping.")
                invalid_videos += 1
                continue

            data_list.append({
                "video_path": expected_video_path,
                "text": text,
                "video_id": video_id
            })
        else:
            print(f"Warning: Video file '{expected_video_filename}' not found for ID '{video_id}'. Skipping.")
            missing_videos += 1

    print(f"Total valid samples built: {len(data_list)}")
    print(f"Samples without video: {missing_videos}")
    print(f"Samples with invalid/empty/long videos: {invalid_videos}")

    return data_list


# --- 3. 数据集类定义 ---
class CSLDailyDataset(Dataset):
    def __init__(self, data_list, max_frames=MAX_FRAMES, target_size=(224, 224), gloss_to_id=None, is_train=False):
        self.data_list = data_list
        self.max_frames = max_frames
        self.target_size = target_size
        self.gloss_to_id = gloss_to_id
        self.is_train = is_train

        # 👉 关键修改：恢复 ToTensor()，并确保其在 transform 的开头
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                # 👈 将 ToTensor() 放在前面，确保后续操作接收的是 Tensor
                transforms.ToTensor(),
                # 👈 其他增强操作现在接收的是 Tensor
                transforms.RandomApply([
                    transforms.RandomResizedCrop(target_size, scale=(0.75, 1.0))
                ], p=0.7),  # 提高概率
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
                ], p=0.5),  # 提高概率
                transforms.RandomApply([
                    # 👉 新增：随机水平翻转（仅对非手语关键帧）
                    transforms.RandomHorizontalFlip(p=0.3)
                ], p=0.3),
                # 👈 移除有风险的 lambda 增强，或用更安全的方式替代
                # transforms.RandomApply([
                #     lambda x: x + torch.randn_like(x) * 0.05
                # ], p=0.2),
                # 👈 最后进行归一化
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        video_path = item.get('video_path', '')
        text = item.get('text', '')

        if isinstance(video_path, list):
            if len(video_path) > 0:
                video_path = video_path[0]
            else:
                raise ValueError(f"Empty list for video_path at index {idx}")
        elif not isinstance(video_path, str) or not video_path:
            raise ValueError(f"Invalid video_path at index {idx}: '{video_path}' (type: {type(video_path)})")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at index {idx}: '{video_path}'")

        # 👇 不再一次性加载所有帧，而是记录路径和元信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Failed to open video file at index {idx}: '{video_path}'. Returning empty frames.")
            processed_frames = torch.empty(0, 3, *self.target_size)
            gloss_tokens = []
            cap.release()
            return processed_frames, gloss_tokens

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 0:
            print(f"Warning: Video file '{video_path}' has no frames for ID '{video_path}'. Skipping.")
            return torch.empty(0, 3, *self.target_size), []

        # 👉 限制最大帧数
        actual_frames = min(total_frames, self.max_frames)

        # 👉 创建一个列表来存储帧索引，用于后续随机采样或均匀采样
        # 如果不需要 temporal_stretch，可以直接用 range(actual_frames)
        frame_indices = list(range(actual_frames))

        # 👉 新增：如果是训练模式，可以应用时间拉伸（通过重采样索引实现）
        if self.is_train:
            stretch_factor = np.random.uniform(0.8, 1.2)
            new_length = int(len(frame_indices) * stretch_factor)
            if new_length > 0:
                indices = np.linspace(0, len(frame_indices) - 1, new_length).astype(int)
                frame_indices = [frame_indices[i] for i in indices]

        # 👉 关键修改：只加载需要的帧
        frames = []
        cap = cv2.VideoCapture(video_path)
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # 设置到指定帧
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 👈 应用 transform 到 numpy 数组
            pil_image = Image.fromarray(frame_rgb)
            transformed_image = self.transform(pil_image)
            frames.append(transformed_image)
        cap.release()

        if len(frames) == 0:
            processed_frames = torch.empty(0, 3, *self.target_size)
        else:
            # 👈 将列表中的 Tensor 堆叠成一个大 Tensor
            processed_frames = torch.stack(frames)

        # 处理文本
        if isinstance(text, list):
            text = " ".join(str(item) for item in text)
        elif not isinstance(text, str):
            text = ""

        text = text.strip()
        gloss_tokens = [token for token in text.split() if token.strip()]

        # 显式删除临时变量
        del frames
        del frame_indices

        return processed_frames, gloss_tokens


# --- 4. 模型定义 ---
class LightweightSLRModel(nn.Module):
    def __init__(self, num_classes, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(LightweightSLRModel, self).__init__()
        from torchvision.models import mobilenet_v2  # 👈 修改：使用MobileNetV2
        self.cnn = mobilenet_v2(weights="IMAGENET1K_V1").features  # 仅特征提取层
        # 👈 关键修改：添加自适应平均池化层，强制将特征图压缩为 (1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 👈 关键修改：移除对 cnn_features 的硬编码，通过一次前向传播确定其实际值
        # 通过一次前向传播确定特征向量的长度
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            cnn_out = self.cnn(dummy_input)
            cnn_out = self.adaptive_pool(cnn_out)
            self.cnn_features = cnn_out.view(1, -1).size(1)  # 👈 动态获取特征长度

        # 👈 修改：增加CNN后Dropout
        self.cnn_dropout = nn.Dropout(dropout * 0.5)  # 较小的dropout

        self.lstm = nn.LSTM(
            input_size=self.cnn_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 👈 修改：单向LSTM，减少参数
        )
        self.lstm_output_size = hidden_size  # 单向LSTM

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.lstm_output_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        # 👈 关键修复：使用自适应池化层将特征图压缩为 (B*T, 1280, 1, 1)
        x = self.adaptive_pool(x)
        # 👈 关键修复：展平空间维度，得到 (B*T, 1280)
        x = torch.flatten(x, start_dim=1)
        # 👈 关键修复：重新 reshape 为 (B, T, 1280)
        x = x.view(B, T, -1)
        x = self.cnn_dropout(x)  # 👈 新增：CNN后Dropout
        lstm_out, _ = self.lstm(x)
        logits = self.classifier(self.dropout(lstm_out))
        return logits  # [B, T, num_classes]

# --- 5. 词汇表构建函数 (新增: 数据清洗) ---
def build_vocabulary(data_list, annotation_field='text'):
    """从数据列表中构建词汇表"""
    unique_glosses = set()
    for item in data_list:
        gloss_str = item.get(annotation_field, '')
        gloss_str = gloss_str.strip()  # 👈 新增: 清除首尾空白
        # 👇 修改: 分割并过滤空字符串
        gloss_tokens = [token for token in gloss_str.split() if token.strip()]
        unique_glosses.update(gloss_tokens)

    print(f"Found {len(unique_glosses)} unique glosses.")

    # 👉 关键修改：确保<blank>在首位
    special_tokens = ["<blank>", "<pad>", "<sos>", "<eos>", "<unk>"]
    all_glosses_list = special_tokens + sorted(list(unique_glosses))

    gloss_to_id = {gloss: idx for idx, gloss in enumerate(all_glosses_list)}
    id_to_gloss = {idx: gloss for gloss, idx in gloss_to_id.items()}

    print("Vocabulary built.")
    return gloss_to_id, id_to_gloss


# --- 6. 自定义批次合并函数 (适配 CTC Loss) ---
def collate_fn(batch, gloss_to_id):
    """自定义批次合并函数，处理不同长度的视频和标签，用于 CTC Loss。"""
    frames_batch = [item[0] for item in batch]
    glosses_batch = [item[1] for item in batch]

    # 填充帧到最大长度
    max_t = max(f.size(0) for f in frames_batch)
    padded_frames_batch = []
    input_lengths = []
    for frames in frames_batch:
        t, c, h, w = frames.size()
        input_lengths.append(t)
        if t < max_t:
            pad_size = max_t - t
            padding = torch.zeros(pad_size, c, h, w, dtype=frames.dtype, device=frames.device)
            frames_padded = torch.cat([frames, padding], dim=0)
        else:
            frames_padded = frames[:max_t]
        padded_frames_batch.append(frames_padded)
    padded_frames_batch = torch.stack(padded_frames_batch)

    # 转换gloss为ID并处理用于 CTC
    targets = []
    target_lengths = []
    for glosses in glosses_batch:
        if not isinstance(glosses, list):
            glosses = []

        # 将gloss字符串列表转换为ID列表，跳过 <pad>
        ids = []
        for gloss in glosses:
            # 👉 使用<blank>而不是<pad>作为CTC的blank
            token_id = gloss_to_id.get(gloss, gloss_to_id.get('<unk>', gloss_to_id['<unk>']))
            if token_id != gloss_to_id.get('<pad>', gloss_to_id['<pad>']):
                ids.append(token_id)
        targets.extend(ids)
        target_lengths.append(len(ids))

    targets = torch.tensor(targets, dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)

    return padded_frames_batch, targets, input_lengths, target_lengths

# --- 7. 标签平滑CTC损失函数 ---
def label_smoothing_ctc_loss(log_probs, targets, input_lengths, target_lengths, smoothing=0.1, blank_idx=0):
    """
    实现带标签平滑的CTC损失
    """
    # 计算原始CTC损失
    ctc_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True, reduction='none')
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

    # 计算标签平滑项
    # 平滑目标：大部分概率分配给正确标签，小部分均匀分配给其他标签
    vocab_size = log_probs.size(-1)
    smooth_target = torch.ones_like(log_probs) / vocab_size  # 均匀分布

    # 👈 修复：创建一个与 log_probs 形状相同的张量来填充目标位置
    # 我们需要将一维的 targets 映射回 [T, B] 的二维结构
    batch_size = log_probs.size(1)
    max_time = log_probs.size(0)

    # 创建一个全零的二维张量 [T, B]
    target_2d = torch.zeros(max_time, batch_size, dtype=torch.long, device=log_probs.device)

    # 根据 target_lengths 将 targets 填入 target_2d
    current_idx = 0
    for i in range(batch_size):
        length = target_lengths[i].item()
        if length > 0:
            target_2d[:length, i] = targets[current_idx:current_idx + length]
            current_idx += length

    # 使用 gather 来正确地在目标位置分配概率
    # 这里我们直接在 [T, B] 上操作，然后扩展到 [T, B, vocab_size]
    # 更简单的方式：对每个时间步，计算该时间步上所有样本的目标位置
    # 我们可以这样操作：对于每个时间步和每个样本，将正确标签的概率设为 (1-smoothing)
    # 我们可以通过以下方式实现：
    # 方法二：逐个样本处理
    # 这种方法虽然效率稍低，但逻辑清晰，不易出错
    # 我们不再使用 scatter_，而是手动构造平滑目标
    # 但这会很慢，所以还是回到 scatter_ 的思路，但确保维度正确

    # 修正后的 scatter_ 用法：
    # 我们需要一个索引张量，其形状为 [T, B, 1]，内容是每个位置的目标标签
    # 我们已经有了 target_2d_expanded
    # 现在，我们用它来更新 smooth_target
    # 但 smooth_target 是 [T, B, vocab_size]，我们需要在第2维（vocab_size 维度）上进行 scatter
    # 正确的做法是：
    # smooth_target.scatter_(2, target_2d_expanded, 1 - smoothing)
    # 但是，这要求 target_2d_expanded 的形状是 [T, B, 1]，并且它的值是类别索引
    # 这正是我们想要的

    # 👈 修复：使用正确的索引进行 scatter
    # 确保 target_2d_expanded 的形状是 [T, B, 1]
    # 然后在第2维进行 scatter
    target_2d_expanded = target_2d.unsqueeze(-1)  # [T, B, 1]
    smooth_target.scatter_(2, target_2d_expanded, 1 - smoothing)

    # 使用交叉熵作为平滑项
    log_probs_flat = log_probs.permute(1, 0, 2).contiguous().view(-1, vocab_size)  # [T*B, vocab_size]
    smooth_target_flat = smooth_target.permute(1, 0, 2).contiguous().view(-1, vocab_size)  # [T*B, vocab_size]
    ce_loss = -torch.sum(smooth_target_flat * log_probs_flat, dim=1)  # [T*B]

    # 平均损失
    ce_loss = ce_loss.mean()

    # 结合CTC损失和平滑项
    combined_loss = (1 - smoothing) * loss.mean() + smoothing * ce_loss

    return combined_loss

# --- 8. 主训练函数 ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        import openpyxl
    except ImportError:
        print("Installing 'openpyxl' library...")
        os.system("pip install openpyxl")
    try:
        import pandas
    except ImportError:
        print("Installing 'pandas' library...")
        os.system("pip install pandas")

    print("\n=== Step 1: Building Local Data List ===")
    # 👈 调用修改后的函数，传入 MAX_FRAMES
    data_list = build_local_data_list(VIDEO_DIR, ANNOTATION_FILE, max_frames=MAX_FRAMES)

    print("\n=== Step 2: Splitting Data into Train/Val ===")
    split_index = int(0.9 * len(data_list))
    train_data_list = data_list[:split_index]
    val_data_list = data_list[split_index:]

    print(f"Training set size: {len(train_data_list)}")
    print(f"Validation set size: {len(val_data_list)}")

    print("\n=== Step 3: Building Vocabulary ===")
    # 👈 使用修改后的 build_vocabulary 函数
    gloss_to_id, id_to_gloss = build_vocabulary(data_list, annotation_field='text')
    vocab_size = len(gloss_to_id)
    print(f"Vocabulary size: {vocab_size}")

    with open('gloss_to_id.json', 'w', encoding='utf-8') as f:
        json.dump(gloss_to_id, f, ensure_ascii=False, indent=2)
    with open('id_to_gloss.json', 'w', encoding='utf-8') as f:
        json.dump(id_to_gloss, f, ensure_ascii=False, indent=2)

    print("\n=== Step 4: Creating Data Loaders ===")
    train_dataset = CSLDailyDataset(data_list=train_data_list, max_frames=MAX_FRAMES, target_size=TARGET_SIZE, gloss_to_id=gloss_to_id, is_train=True)
    val_dataset = CSLDailyDataset(data_list=val_data_list, max_frames=MAX_FRAMES, target_size=TARGET_SIZE, gloss_to_id=gloss_to_id, is_train=False)

    def train_collate_fn(batch):
        return collate_fn(batch, gloss_to_id)

    def val_collate_fn(batch):
        return collate_fn(batch, gloss_to_id)

    # 👈 关键修改：增加 prefetch_factor 以提高数据预取效率
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_collate_fn, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=val_collate_fn, prefetch_factor=2)

    print("\n=== Step 5: Initializing Model ===")
    model = LightweightSLRModel(num_classes=vocab_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT)
    model.to(device)

    # 👉 关键修改：确保 blank_idx 为0（<blank>的索引）
    blank_idx = gloss_to_id['<blank>']
    assert blank_idx == 0, f"Expected blank_idx to be 0, got {blank_idx}. Check vocabulary order."

    # 👈 修改：使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 👈 新增：学习率调度器 - 调整参数
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,  # 👈 现在使用新的NUM_EPOCHS=30
        pct_start=0.2,  # 👈 增加预热阶段到20%
        anneal_strategy='cos',
        div_factor=25.0,  # 👈 增大初始学习率与最大学习率的比值
        final_div_factor=1e4  # 👈 最终学习率更小
    )

    print("\n=== Step 6: Starting Training ===")
    model.train()
    train_losses = []
    val_losses = []

    # 👈 新增：早停机制
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NUM_EPOCHS): # 👈 现在是30轮
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")

        for batch_idx, (batch_frames, batch_targets, batch_input_lengths, batch_target_lengths) in enumerate(progress_bar):
            batch_frames = batch_frames.to(device)
            batch_targets = batch_targets.to(device)
            batch_input_lengths = batch_input_lengths.to(device)
            batch_target_lengths = batch_target_lengths.to(device)

            optimizer.zero_grad()

            outputs = model(batch_frames)  # [B, T, num_classes]
            log_probs = torch.log_softmax(outputs, dim=2).permute(1, 0, 2)  # [T, B, num_classes]

            # 👈 使用标签平滑CTC损失
            loss = label_smoothing_ctc_loss(
                log_probs,
                batch_targets,
                batch_input_lengths,
                batch_target_lengths,
                smoothing=0.1,  # 标签平滑参数
                blank_idx=blank_idx
            )

            loss.backward()

            # 👈 新增：梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()  # 👈 新增：调度器步进

            loss_value = loss.item()
            running_loss += loss_value
            progress_bar.set_postfix(loss=loss_value)

        avg_loss = running_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")
        train_losses.append(avg_loss)

        # --- 验证 ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc="Validation")
            for val_batch_idx, (val_batch_frames, val_batch_targets, val_batch_input_lengths, val_batch_target_lengths) in enumerate(val_progress_bar):
                val_batch_frames = val_batch_frames.to(device)
                val_batch_targets = val_batch_targets.to(device)
                val_batch_input_lengths = val_batch_input_lengths.to(device)
                val_batch_target_lengths = val_batch_target_lengths.to(device)

                val_outputs = model(val_batch_frames)  # [B, T, num_classes]
                val_log_probs = torch.log_softmax(val_outputs, dim=2).permute(1, 0, 2)  # [T, B, num_classes]

                # 验证时也使用标签平滑损失，但可以不使用标签平滑以获得真实性能
                val_loss = nn.CTCLoss(blank=blank_idx, zero_infinity=True, reduction='mean')(
                    val_log_probs, val_batch_targets, val_batch_input_lengths, val_batch_target_lengths
                )

                val_running_loss += val_loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        val_losses.append(avg_val_loss)

        # 👈 改进的早停检查
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
            }, 'best_model.pth')
            print(f"New best model saved with validation loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs. No improvement for {EARLY_STOP_PATIENCE} epochs.")
                break

        # 👉 额外检查：如果训练损失远低于验证损失且验证损失不再下降，提前停止
        if avg_loss < 0.5 and avg_val_loss > 3.0 and epoch > 15: # 15是30的一半
            print(f"Training loss too low ({avg_loss}) compared to validation loss ({avg_val_loss}). Possible overfitting. Stopping early.")
            break

        model.train()

    print("\n=== Step 7: Plotting and Saving ===")
    import numpy as np
    train_losses_np = np.array(train_losses, dtype=np.float32)
    val_losses_np = np.array(val_losses, dtype=np.float32)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses_np) + 1), train_losses_np, label='Training Loss', marker='o')
    if len(val_losses_np) > 0:
        plt.plot(range(1, len(val_losses_np) + 1), val_losses_np, label='Validation Loss', marker='s')
    plt.title('Model Training Loss (CTC with Label Smoothing) - 30 Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_loss_plot_ctc_fixed.png')
    plt.show()

    # 保存最终模型
    torch.save(model.state_dict(), 'slr_model_ctc_fixed_final.pth')
    print("Final model saved as 'slr_model_ctc_fixed_final.pth'.")
    print(f"Best model saved as 'best_model.pth' with validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory '{VIDEO_DIR}' does not exist.")
        print("Please create a folder named 'videos' in the project root and place your .mp4 files inside it.")
        exit(1)

    if not os.path.exists(ANNOTATION_FILE):
        print(f"Error: Annotation file '{ANNOTATION_FILE}' does not exist.")
        print("Please ensure the Excel file is present.")
        exit(1)

    local_dataset_dir = "local_csl_daily"
    if os.path.exists(local_dataset_dir):
        print(f"Removing old dataset cache: {local_dataset_dir}")
        import shutil
        shutil.rmtree(local_dataset_dir)

    train_model()