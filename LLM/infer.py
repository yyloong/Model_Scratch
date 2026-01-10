from dist_train import load_checkpoint
import torch 
from Model.mini_llm import MiniLLM,MiniLLMConfig
from transformers import AutoTokenizer


def run_inference_test():
    """
    运行推理测试的主函数。
    """
    # 1. 初始化模型配置
    config = MiniLLMConfig()

    # 2. 实例化模型
    model = MiniLLM(config)
    
    # 3. 定义 checkpoint 路径和设备
    model_name = "Qwen/Qwen3-0.6B"
    checkpoint_path = '/home/u-longyy/data/mini_model/ds_checkpoints/pytorch_model.bin'  # <--- 修改为你的 checkpoint 路径
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4. 加载 checkpoint
    try:
        model_dict = torch.load(checkpoint_path)
        #model = torch.compile(model)
        model.load_state_dict(model_dict)
        model.to(dtype=torch.bfloat16)
    except FileNotFoundError:
        print(f"错误：Checkpoint 文件未找到，请在 '{checkpoint_path}' 放置你的模型文件。")
        print("正在使用随机初始化的模型进行演示。")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # 5. 准备输入数据
    # 这里的输入是一个 token ID 的序列，你需要根据你的 tokenizer 来生成
    # 假设 151643 是起始符 <bos> 的 ID
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_text = input("请输入文本:")
    input_ids = tokenizer(input_text,return_tensors='pt')["input_ids"][0]
    input_ids = torch.cat([torch.tensor([151644],dtype=torch.int32),input_ids]).unsqueeze(0).to("cuda:0")

    # 6. 运行生成过程
    print("\n开始生成文本...")
    # 使用 torch.no_grad() 来禁用梯度计算，可以节省显存并加速推理
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=1024,      # 生成的最大长度
            temperature=0.7,   # 控制生成文本的随机性
        )
    
    print("生成完成！")
    
    # 7. 解码并打印结果
    # 你需要一个 tokenizer 来将 token ID 解码回文本
    # 这里我们只打印生成的 token ID 序列
    print(f"生成的 Token IDs: {generated_ids.tolist()}")
    text = tokenizer.decode(generated_ids[0])
    print(f"生成的文本: {text}")
    
    # 示例：如果你有一个 tokenizer
    # tokenizer = YourTokenizer() # 加载你的 tokenizer
    # generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # print(f"\n生成的文本:\n{generated_text}")


if __name__ == '__main__':
    run_inference_test()