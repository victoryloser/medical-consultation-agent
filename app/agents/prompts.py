SYMPTOM_EXTRACT_PROMPT = """你是医疗助理，请从以下用户描述中提取症状列表，以JSON数组格式输出，不要解释，只输出JSON数组。
例如：["发热","咳嗽","乏力"]
如果没有明显症状，输出：[]

用户描述：{context}"""

ADVICE_GENERATE_PROMPT = """你是医疗健康咨询助手，请根据以下信息生成健康咨询建议。

症状：{symptoms}
风险等级：{risk_level}
图像分析：{image_analysis}

参考医学知识：
{retrieved_docs}

请用Markdown格式输出以下内容：
1. 可能相关情况（不要给出确诊结论）
2. 建议就医科室
3. 建议检查方向
4. 生活方式建议

注意：不要给出处方药具体剂量，不要替代医生判断，不要做确诊结论。"""

RISK_ASSESS_PROMPT = """你是医疗风险评估助手，请根据以下症状和参考知识评估风险等级。

症状：{symptoms}
参考知识要点：{knowledge_summary}

请只输出一个风险等级，从以下选择：low、medium、high
（emergency 级别由急症规则引擎负责，不在此处输出）
只输出一个词，不要解释。"""

EMERGENCY_CONFIRM_PROMPT = """你是急诊分诊助手，请判断以下描述是否属于需要立即急诊的紧急情况。

用户描述：{text}
初步匹配场景：{category}

判断标准：只有存在生命危险或需要立即急救的情况才回答 yes。
仅输出 yes 或 no，不要解释。"""

# ── 三层验证 Prompt（参照 WSI-Agents 验证机制）────────────────────────────────

FACT_CHECK_PROMPT = """你是医疗知识核查助手。请判断以下建议内容是否与提供的医学参考知识一致。

建议内容：
{advice}

医学参考知识：
{knowledge}

请只输出一个0到1之间的小数，表示一致性分数：
- 1.0 表示完全一致，无矛盾
- 0.5 表示部分一致，有小偏差
- 0.0 表示存在明显矛盾
只输出数字，不要解释。"""

CONSENSUS_PROMPT = """你是医疗风险评估助手。根据以下症状，评估风险等级。

症状：{symptoms}
用户描述：{user_text}

只输出一个词：low、medium、high 或 emergency，不要解释。"""
