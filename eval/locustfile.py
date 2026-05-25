"""
维度三：系统性能压测
使用 locust 对 POST /api/consultation 接口进行并发压测。

安装：pip install locust
用法（命令行无头模式，推荐）：
    locust -f eval/locustfile.py --headless \
           -u 5 -r 1 --run-time 60s \
           --host http://localhost:8000 \
           --html eval/report.html

参数说明：
    -u   并发用户数（本地 Ollama 建议 3~5，API 模式可调高）
    -r   每秒新增用户数（ramp-up 速率）
    --run-time  持续时间
    --html      输出 HTML 报告

查看指标：
    - Requests/s（RPS）
    - 响应时间 P50 / P90 / P95 / P99
    - 失败率（Failure %）
"""
import random

from locust import HttpUser, between, task

# 覆盖不同风险等级的典型请求
PAYLOADS = [
    # low
    "我发烧 37.5 度，有点乏力，睡眠不好",
    "喉咙有点不舒服，轻微咳嗽两天",
    "眼睛有点干，总想揉眼睛",
    # medium
    "咳嗽持续三周了，有时候有黄痰",
    "发烧 38.8 度，咳嗽，乏力，持续三天了",
    "皮肤出现大片红疹，有点痒，不知道是什么原因",
    # high
    "高烧 39.5 度三天不退，吃了退烧药也不管用",
    "腹痛很厉害，持续好几个小时了，弯不下腰",
    # emergency
    "胸口突然很痛，冒冷汗，左肩也疼，感觉很难受",
    "突然右侧肢体无力，说话不清楚，嘴歪了",
]


class ConsultationUser(HttpUser):
    # 每次请求后等待 1~3 秒（模拟真实用户思考时间）
    wait_time = between(1, 3)

    @task(8)
    def consult_text_only(self):
        """纯文字问诊（主场景，权重 8）"""
        text = random.choice(PAYLOADS)
        with self.client.post(
            "/api/consultation",
            data={"text": text, "model_provider": "auto"},
            catch_response=True,
            name="POST /api/consultation [text]",
        ) as resp:
            if resp.status_code == 200:
                body = resp.json()
                if not body.get("final_report"):
                    resp.failure("响应缺少 final_report 字段")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @task(2)
    def health_check(self):
        """健康检查接口（权重 2，基线对比用）"""
        with self.client.get(
            "/api/health",
            catch_response=True,
            name="GET /api/health",
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")
