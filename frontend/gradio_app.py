import gradio as gr
import requests

API_URL = "http://localhost:8000/api/consultation"

RISK_STYLE = {
    "low": ("低风险", "#2e7d32"),
    "medium": ("中风险", "#e65100"),
    "high": ("高风险", "#c62828"),
    "emergency": ("急症风险 ⚠️", "#b71c1c"),
    "unknown": ("未知", "#757575"),
}


def consult(text: str, image, model_provider: str):
    if not text or not text.strip():
        return "请输入症状描述后再发起问诊。", "", ""

    data = {"text": text.strip(), "model_provider": model_provider}
    files = {}

    try:
        if image is not None:
            with open(image, "rb") as f:
                image_bytes = f.read()
            files["image"] = ("image.jpg", image_bytes, "image/jpeg")

        response = requests.post(
            API_URL,
            data=data,
            files=files if files else None,
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

        risk_key = result.get("risk_level", "unknown")
        risk_label, risk_color = RISK_STYLE.get(risk_key, RISK_STYLE["unknown"])
        risk_html = (
            f'<div style="padding:12px;border-radius:8px;background:{risk_color}20;'
            f'border-left:4px solid {risk_color}">'
            f'<span style="color:{risk_color};font-size:1.2em;font-weight:bold">{risk_label}</span>'
            f"</div>"
        )

        refs = result.get("references", [])
        refs_md = "**参考知识来源：**\n" + (
            "\n".join(f"- {r['title']} （{r['source']}）" for r in refs)
            if refs
            else "无"
        )

        report = result.get("final_report", "未能生成报告")
        return report, risk_html, refs_md

    except requests.exceptions.ConnectionError:
        return (
            "无法连接后端服务，请确认 FastAPI 已启动：\n```\nuvicorn app.main:app --host 0.0.0.0 --port 8000\n```",
            "",
            "",
        )
    except requests.exceptions.Timeout:
        return "请求超时（120s），请稍后重试或检查模型是否正常运行。", "", ""
    except Exception as e:
        return f"请求失败：{e}", "", ""


EXAMPLE_INPUTS = [
    ["我发烧 38.5 度，咳嗽两天，还有点乏力，需要去医院吗？", None, "auto"],
    ["胸口突然很痛，还冒冷汗，左肩也疼，很难受。", None, "auto"],
    ["身上起了很多红疹子，又痒又难受，不知道是什么。", None, "auto"],
    ["我最近总是头晕，血压有点高，140/90，该怎么办？", None, "auto"],
]

with gr.Blocks(title="多模态医疗辅助问诊 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 多 Agent 多模态医疗辅助问诊 Demo

> **声明：** 本系统仅用于学习演示，**不能替代医生诊断**，不提供处方，不构成医疗建议。如有急症，请立即拨打 **120**。
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="症状描述",
                lines=6,
                placeholder="请描述您的症状、持续时间、严重程度等\n例如：发烧 38.5 度，咳嗽两天，有点乏力",
            )
            image_input = gr.Image(label="上传图片（可选，如皮疹、外伤等）", type="filepath")
            provider_radio = gr.Radio(
                choices=["auto", "ollama", "api"],
                value="auto",
                label="模型调用方式",
                info="auto：按配置文件自动选择；ollama：本地模型；api：云端 API",
            )
            submit_btn = gr.Button("开始问诊", variant="primary", size="lg")

        with gr.Column(scale=2):
            risk_output = gr.HTML(label="风险等级")
            report_output = gr.Markdown(label="问诊报告")
            refs_output = gr.Markdown(label="参考来源")

    gr.Examples(
        examples=EXAMPLE_INPUTS,
        inputs=[text_input, image_input, provider_radio],
        label="示例输入",
    )

    submit_btn.click(
        fn=consult,
        inputs=[text_input, image_input, provider_radio],
        outputs=[report_output, risk_output, refs_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
