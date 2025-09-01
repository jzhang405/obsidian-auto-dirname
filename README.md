# Obsidian Auto Directory Name

使用大语言模型（LLM）自动生成 Obsidian 笔记目录结构的 Python 工具。

## 功能特点

- 🤖 **AI 驱动**：使用 LLM 根据关键词和语言环境智能生成目录结构
- 📁 **多种组织方法**：支持 PARA、Zettelkasten 和默认组织方法
- 🌏 **多语言支持**：支持中文和英文 locale
- 🔢 **数量控制**：可设置生成目录的最小和最大数量（默认 5-10 个）
- 👀 **预览模式**：支持 dry-run 模式预览变更
- 🔄 **目录融合**：保留现有目录并智能整合新生成的结构
- 🎨 **可视化预览**：生成彩色树形目录预览（新增为绿色）
- 💾 **结构保存**：将目录结构保存为 JSON 文件
- 📂 **自动分类**：自动生成"未分类"目录处理难以分类的内容

## 安装

1. **克隆项目**：
   ```bash
   git clone https://github.com/jzhang405/obsidian-auto-dirname.git
   cd obsidian-auto-dirname
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

## LLM 配置

使用 LLM 功能需要在 `.env` 文件中配置你的 LLM 提供商和 API 密钥。

### 配置步骤

1. **复制配置模板**：
   ```bash
   cp .env.example .env
   ```

2. **编辑 `.env` 文件**，使用 **统一 LLM 配置**（推荐）：
   ```bash
   # 统一 LLM 配置（推荐）
   LLM_PROVIDER=deepseek
   LLM_API_KEY=your_api_key_here
   LLM_BASE_URL=https://api.deepseek.com/v1
   LLM_MODEL=deepseek-chat
   
   # LLM 参数
   LLM_MAX_TOKENS=500
   LLM_TEMPERATURE=0.7
   ```

### 支持的 LLM 提供商

| 提供商 | `LLM_PROVIDER` | 推荐模型 | API 获取 |
|--------|----------------|------------|----------|
| **DeepSeek** | `deepseek` | `deepseek-chat` | [DeepSeek 控制台](https://platform.deepseek.com/) |
| **OpenAI** | `openai` | `gpt-3.5-turbo-instruct` | [OpenAI 控制台](https://platform.openai.com/api-keys) |
| **自定义** | `custom` | 自定义模型 | 自定义接口 |

### 配置优先级

✅ **首选**：统一配置 (`LLM_API_KEY`, `LLM_BASE_URL`)  
🔄 **备选**：特定提供商配置 (`DEEPSEEK_API_KEY`, `OPENAI_API_KEY`)

> **提示**：如果未配置 API 密钥，程序将使用占位符响应。

## 使用方法

### 基本命令

```bash
python obsidian_auto_dirname.py [options]
```

### 命令行选项

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input-dir` | TEXT | `.` | 输入目录路径（默认：当前目录） |
| `--locale` | TEXT | `en` | 目录名称语言（`en` 或 `zh`） |
| `--keywords` | TEXT | - | 逗号分隔的关键词列表 |
| `--method` | CHOICE | `default` | 组织方法（`PARA`、`Zettelkasten`、`default`） |
| `--min` | INTEGER | `5` | 生成目录的最小数量 |
| `--max` | INTEGER | `10` | 生成目录的最大数量 |
| `--dry-run` | FLAG | - | 预览模式，不实际创建目录 |
| `--verbose` | FLAG | - | 详细模式，显示调试信息 |
| `--help` | FLAG | - | 显示帮助信息 |

## 目录结构规范

### 目录层级
- **严格二级结构**：主目录 + 子目录，不允许三级或更深层目录
- **特例**：PARA 方法可以生成三级目录结构

### 命名规范
- **主目录格式**：`01.DirectoryName`（数字 + 点号 + 名称，无空格）
- **子目录格式**：简单名称，无数字前缀
- **编号规则**：从 `01` 开始，优先级越高数字越小

### 目录过滤
- **识别规则**：只识别符合命名规范的现有目录
- **支持格式**：`01.Work`、`01. Work`、`01 - Work`
- **忽略目录**：不符合规范的目录和隐藏目录

### 自动生成
- **未分类目录**：自动生成，中文为"未分类"，英文为"Uncategorized"

## 使用示例

### 1. 基本使用
```bash
# 使用默认设置生成目录
python obsidian_auto_dirname.py

# 预览模式，不实际创建
python obsidian_auto_dirname.py --dry-run
```

### 2. 关键词驱动生成
```bash
# 英文关键词
python obsidian_auto_dirname.py --keywords="programming,design,marketing"

# 中文关键词 + 中文 locale
python obsidian_auto_dirname.py --locale=zh --keywords="编程,设计,市场营销"
```

### 3. 组织方法使用
```bash
# 使用 PARA 方法（支持三级目录）
python obsidian_auto_dirname.py --method=PARA --keywords="项目,领域,资源,归档"

# 使用 Zettelkasten 方法
python obsidian_auto_dirname.py --method=Zettelkasten --keywords="概念,链接,索引"
```

### 4. 数量控制
```bash
# 设置生成 3-6 个目录
python obsidian_auto_dirname.py --min=3 --max=6 --keywords="工作,学习,生活"

# 显示详细调试信息
python obsidian_auto_dirname.py --verbose --dry-run --keywords="测试"
```

### 5. 自定义目录和预览
```bash
# 指定目录位置
python obsidian_auto_dirname.py --input-dir="/path/to/obsidian/vault" --dry-run

# 全功能示例
python obsidian_auto_dirname.py \
  --input-dir="./my-vault" \
  --locale=zh \
  --keywords="工作,学习,生活,项目" \
  --method=default \
  --min=4 --max=8 \
  --verbose \
  --dry-run
```

## 输出示例

### 目录结构预览
```
Directory Structure Preview for '/path/to/vault':
├── 01.工作 (new)
│   └── 项目管理 (new)
│   └── 会议记录 (new)
│   └── 职业发展 (new)
├── 02.学习 (new)
│   └── 课程笔记 (new)
│   └── 读书笔记 (new)
│   └── 研究资料 (new)
├── 03.生活 (new)
│   └── 健康管理 (new)
│   └── 财务管理 (new)
└── 04.未分类 (new)
```

### JSON 输出文件
```json
{
  "categories": [
    {
      "path": "01.工作",
      "subcategories": ["项目管理", "会议记录", "职业发展"]
    },
    {
      "path": "02.学习",
      "subcategories": ["课程笔记", "读书笔记", "研究资料"]
    }
  ]
}
```

## 高级功能

### 现有目录融合
- 自动识别和保留符合命名规范的现有目录
- 为现有目录添加适合的子目录
- 智能避免重复和冲突

### 多语言支持
- **中文 (zh)**：生成中文目录名和子目录
- **英文 (en)**：生成英文目录名和子目录
- 未分类目录自动适配语言

### 组织方法特色
- **PARA**：支持项目、领域、资源、归档的三级结构
- **Zettelkasten**：适用于知识卡片系统的目录结构
- **Default**：通用的二级目录结构

## 注意事项

- 🔒 **API 密钥安全**：请妥善保管你的 LLM API 密钥，不要将其提交到版本控制系统
- 📁 **目录安全**：程序不会删除现有目录，只会添加和整合
- 🔎 **预览模式**：建议先使用 `--dry-run` 查看效果
- 🎨 **命名规范**：保持一致的目录命名格式以获得最佳效果

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License