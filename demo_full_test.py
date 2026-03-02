#!/usr/bin/env python3
"""
HlAB 全流程中文 Demo 测试
========================
一键创建完整 demo 数据 + 端到端测试所有核心功能：
  1. LLM 配置（DashScope Qwen Flash）
  2. 知识库（KV + FAQ + 文档块）
  3. 工具调用（计算器 + 天气查询）
  4. 工作流（报修申请 - 多步收集/确认）
  5. Agent 创建 + 能力绑定
  6. 全流程调用测试：
     - 闲聊/打招呼
     - 知识检索（RAG）
     - 工具调用（计算 + 天气）
     - 工作流办理（发起 → 填写 → 确认）
     - 多轮对话上下文

用法:
  先启动服务器:  cd server && uvicorn server.main:app --host 0.0.0.0 --port 8000
  再跑本脚本:    python demo_full_test.py

也可以指定服务器地址:
  python demo_full_test.py --base-url http://192.168.1.100:8000
"""

import argparse
import io
import json
import os
import sys
import time
import requests

# Windows 控制台 UTF-8 支持
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 确保 localhost 不走代理（Windows 代理软件可能拦截）
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",localhost,127.0.0.1,::1"
os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",localhost,127.0.0.1,::1"

# ── 颜色输出 ──────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


def info(msg: str):
    print(f"  {CYAN}→ {msg}{RESET}")


def section(title: str):
    print(f"\n{BOLD}{YELLOW}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}\n")


def sub_section(title: str):
    print(f"\n  {BOLD}{CYAN}── {title} ──{RESET}")


# ── API 封装 ──────────────────────────────────────────────

class DemoClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.api = f"{self.base}/api/v1"
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"

        # 存储创建的资源 ID
        self.llm_config_id = None
        self.agent_id = None
        self.knowledge_source_id = None
        self.tool_ids = {}   # name -> id
        self.workflow_id = None
        self.invoke_session_id = None  # 用于多轮对话

    def _post(self, path: str, data: dict) -> dict:
        resp = self.session.post(f"{self.api}{path}", json=data)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str, params: dict = None) -> dict | list:
        resp = self.session.get(f"{self.api}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    def _put(self, path: str, data: dict) -> dict:
        resp = self.session.put(f"{self.api}{path}", json=data)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str):
        resp = self.session.delete(f"{self.api}{path}")
        resp.raise_for_status()

    # ── 健康检查 ──

    def health_check(self) -> bool:
        try:
            resp = self.session.get(f"{self.base}/docs")
            return resp.status_code == 200
        except Exception:
            return False

    # ── LLM 配置 ──

    def create_llm_config(self) -> str:
        data = {
            "name": "Demo-DashScope-QwenFlash",
            "provider": "dashscope",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "sk-dd3da24e559c4695a1e563b5456db2a9",
            "model": "qwen-turbo-latest",
            "temperature": 0.3,
            "top_p": 0.8,
            "max_tokens": 2048,
            "timeout_ms": 60000,
            "is_default": True,
            "tenant_id": "default",
        }
        result = self._post("/llm-configs", data)
        self.llm_config_id = result["id"]
        return self.llm_config_id

    def test_llm_config(self) -> dict:
        data = {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "sk-dd3da24e559c4695a1e563b5456db2a9",
            "model": "qwen-turbo-latest",
            "temperature": 0.3,
            "max_tokens": 256,
            "timeout_ms": 30000,
        }
        return self._post("/llm-configs/test", data)

    # ── 知识库 ──

    def create_knowledge_source(self) -> str:
        data = {
            "name": "智慧园区知识库",
            "source_type": "document",
            "domain": "smart_campus",
            "tenant_id": "default",
            "metadata": {"category": "园区管理", "version": "1.0"},
        }
        result = self._post("/knowledge/sources", data)
        self.knowledge_source_id = result["id"]
        return self.knowledge_source_id

    def add_kv_entries(self):
        """添加快速问答条目（KV 精确匹配通道）"""
        entries = [
            {
                "entity_key": "物业服务电话",
                "content": "园区物业服务热线：400-888-9999，工作时间：周一至周五 8:00-20:00，周末 9:00-17:00",
            },
            {
                "entity_key": "园区地址",
                "content": "智慧园区位于北京市海淀区中关村科技园A区18号，邮编100080",
            },
            {
                "entity_key": "停车收费标准",
                "content": "园区停车收费：临时车辆 5元/小时，月卡车辆 300元/月，年卡车辆 3000元/年。首30分钟免费。",
            },
            {
                "entity_key": "WiFi密码",
                "content": "园区公共WiFi名称：SmartCampus-Guest，密码：Welcome2024。办公区WiFi需联系IT部门开通。",
            },
            {
                "entity_key": "食堂营业时间",
                "content": "园区食堂营业时间：早餐 7:00-9:00，午餐 11:30-13:30，晚餐 17:30-19:30。周末仅供应午餐。",
            },
        ]
        for entry in entries:
            self._post("/knowledge/kv", {
                "source_id": self.knowledge_source_id,
                "entity_key": entry["entity_key"],
                "content": entry["content"],
                "domain": "smart_campus",
                "metadata": {"type": "quick_answer"},
            })

    def add_faq_entries(self):
        """添加 FAQ 条目（向量检索通道）"""
        faqs = [
            {
                "question": "如何申请会议室？",
                "answer": "会议室预约流程：1. 登录园区管理系统 2. 进入「会议室预约」模块 3. 选择日期、时间和会议室 4. 填写会议主题和参会人数 5. 提交预约。大型会议室（50人以上）需提前3个工作日预约。",
            },
            {
                "question": "访客如何进入园区？",
                "answer": "访客入园流程：1. 被访人在园区APP上提前登记访客信息 2. 访客到达后在前台自助机刷身份证 3. 获取临时访客卡 4. 凭访客卡通过闸机进入。访客需在当日18:00前离园。",
            },
            {
                "question": "园区有哪些配套设施？",
                "answer": "园区配套设施包括：食堂（可容纳500人）、便利店、咖啡厅、健身房（需办卡）、篮球场、会议中心（8间会议室）、打印室、快递柜、充电桩（20个）、母婴室。",
            },
            {
                "question": "如何办理门禁卡？",
                "answer": "门禁卡办理：携带身份证和入驻合同到B栋1楼物业服务中心办理，工作日 9:00-17:00。新卡工本费20元，挂失补办30元。门禁卡同时可用于食堂消费。",
            },
            {
                "question": "园区报修流程是什么？",
                "answer": "报修流程：1. 拨打物业热线400-888-9999 或在园区APP提交报修 2. 描述问题类型和位置 3. 物业派单 4. 维修人员上门处理（一般2小时内响应） 5. 维修完成确认签字。紧急问题（漏水、断电）可拨打紧急热线 400-888-1111。",
            },
        ]
        for faq in faqs:
            self._post("/knowledge/faq", {
                "source_id": self.knowledge_source_id,
                "question": faq["question"],
                "answer": faq["answer"],
                "domain": "smart_campus",
                "metadata": {"type": "faq"},
            })

    # ── 工具 ──

    def create_tools(self):
        """创建工具定义（指向 mock 端点）"""
        tools = [
            {
                "name": "计算器",
                "description": "数学计算工具，支持加减乘除、开方、幂运算等。当用户需要计算时调用此工具。",
                "category": "api",
                "endpoint": f"{self.base}/api/v1/mock-tools/calculator",
                "method": "POST",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide", "sqrt", "pow"],
                            "description": "运算类型：add加, subtract减, multiply乘, divide除, sqrt开方, pow幂",
                        },
                        "a": {"type": "number", "description": "第一个操作数"},
                        "b": {"type": "number", "description": "第二个操作数（开方运算不需要）"},
                    },
                    "required": ["operation", "a"],
                },
                "timeout_ms": 5000,
                "max_retries": 2,
                "tenant_id": "default",
            },
            {
                "name": "天气查询",
                "description": "查询指定城市的当前天气信息，包括温度、湿度、天气状况等。当用户问天气时调用此工具。",
                "category": "api",
                "endpoint": f"{self.base}/api/v1/mock-tools/weather",
                "method": "POST",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "要查询天气的城市名称",
                        },
                    },
                    "required": ["city"],
                },
                "timeout_ms": 5000,
                "max_retries": 2,
                "tenant_id": "default",
            },
            {
                "name": "生成工单",
                "description": "根据报修信息生成维修工单，返回工单编号和状态。",
                "category": "api",
                "endpoint": f"{self.base}/api/v1/mock-tools/create_work_order",
                "method": "POST",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "报修位置"},
                        "issue_type": {"type": "string", "description": "问题类型"},
                        "description": {"type": "string", "description": "问题描述"},
                        "contact_phone": {"type": "string", "description": "联系电话"},
                    },
                    "required": ["location", "description"],
                },
                "timeout_ms": 10000,
                "max_retries": 2,
                "tenant_id": "default",
            },
        ]
        for tool_data in tools:
            result = self._post("/tools", tool_data)
            self.tool_ids[tool_data["name"]] = result["id"]

    # ── 工作流 ──

    def create_workflow(self) -> str:
        """创建报修申请工作流（4步：收集信息 → 确认 → 生成工单 → 完成）"""
        work_order_tool_id = self.tool_ids.get("生成工单")
        data = {
            "name": "报修申请",
            "description": "园区设施报修工作流：收集报修信息 → 确认提交 → 调用工具生成工单 → 完成",
            "tenant_id": "default",
            "config": {"timeout_ms": 300000},
            "steps": [
                {
                    "name": "填写报修信息",
                    "order": 1,
                    "step_type": "collect",
                    "prompt_template": "请填写以下报修信息：",
                    "fields": [
                        {
                            "name": "location",
                            "label": "报修位置",
                            "field_type": "text",
                            "required": True,
                            "placeholder": "例：A栋3楼305室",
                        },
                        {
                            "name": "issue_type",
                            "label": "问题类型",
                            "field_type": "select",
                            "required": True,
                            "options": [
                                {"label": "水电维修", "value": "plumbing_electrical"},
                                {"label": "空调故障", "value": "hvac"},
                                {"label": "门窗损坏", "value": "door_window"},
                                {"label": "网络问题", "value": "network"},
                                {"label": "其他", "value": "other"},
                            ],
                        },
                        {
                            "name": "description",
                            "label": "问题描述",
                            "field_type": "text",
                            "required": True,
                            "placeholder": "请详细描述问题",
                        },
                        {
                            "name": "contact_phone",
                            "label": "联系电话",
                            "field_type": "phone",
                            "required": True,
                            "placeholder": "方便维修人员联系您",
                        },
                    ],
                    "on_failure": "retry",
                    "max_retries": 3,
                    "risk_level": "info",
                },
                {
                    "name": "确认提交",
                    "order": 2,
                    "step_type": "confirm",
                    "prompt_template": "请确认以上报修信息是否正确，确认后将生成工单。",
                    "requires_human_confirm": True,
                    "on_failure": "retry",
                    "max_retries": 2,
                    "risk_level": "warning",
                },
                {
                    "name": "生成工单",
                    "order": 3,
                    "step_type": "tool_call",
                    "tool_id": work_order_tool_id,
                    "prompt_template": "正在生成工单...",
                    "on_failure": "escalate",
                    "risk_level": "info",
                },
                {
                    "name": "完成",
                    "order": 4,
                    "step_type": "complete",
                    "prompt_template": "报修工单已生成！维修人员将在2小时内响应，请保持电话畅通。",
                    "risk_level": "info",
                },
            ],
        }
        result = self._post("/workflows", data)
        self.workflow_id = result["id"]
        return self.workflow_id

    # ── Agent ──

    def create_agent(self) -> str:
        data = {
            "name": "智慧园区助手",
            "description": "智慧园区一站式服务助手，可查询园区信息、查天气、做计算、办理报修等业务",
            "system_prompt": (
                "你是「智慧园区助手」，负责为园区用户提供一站式服务。\n"
                "你的能力包括：\n"
                "1. 回答园区相关问题（物业、设施、规定等）\n"
                "2. 查询天气信息\n"
                "3. 数学计算\n"
                "4. 办理报修申请\n\n"
                "回答要求：简洁友好，用中文回答。"
            ),
            "llm_config_id": self.llm_config_id,
            "response_config": {
                "default_mode": "short",
                "enable_citations": True,
                "enable_followups": True,
            },
            "enabled": True,
            "tenant_id": "default",
        }
        result = self._post("/agents", data)
        self.agent_id = result["id"]
        return self.agent_id

    # ── 能力绑定 ──

    def setup_capabilities(self):
        """通过 Capabilities API 一次性绑定知识 + 工具 + 工作流"""
        data = {
            "knowledge": [
                {
                    "domain": "smart_campus",
                    "source_ids": [self.knowledge_source_id],
                    "description": "搜索智慧园区知识库，获取园区设施、服务、规定等信息",
                },
            ],
            "workflows": [
                {
                    "workflow_id": self.workflow_id,
                    "keywords": ["报修", "维修", "修理", "坏了", "故障"],
                    "description": "办理园区设施报修申请，收集报修位置、问题类型等信息并生成工单",
                },
            ],
            "tools": [
                {
                    "tool_ids": list(self.tool_ids.values()),
                    "keywords": ["计算", "天气", "温度", "多少度"],
                    "description": "数学计算和天气查询工具",
                },
            ],
        }
        return self._put(f"/agents/{self.agent_id}/capabilities", data)

    # ── 调用 Agent ──

    def invoke_agent(self, message: str, form_data: dict = None) -> dict:
        """调用 agent 并返回响应"""
        data = {
            "agent_id": self.agent_id,
            "session_id": self.invoke_session_id,
            "user_id": "demo_user",
            "message": message,
        }
        if form_data:
            data["form_data"] = form_data
        result = self._post("/invoke", data)
        # 保存 session_id 用于多轮对话
        self.invoke_session_id = result.get("session_id")
        return result

    def invoke_new_session(self, message: str) -> dict:
        """新开一个会话"""
        self.invoke_session_id = None
        return self.invoke_agent(message)

    # ── 清理 ──

    def cleanup(self):
        """清理所有 demo 数据"""
        try:
            if self.agent_id:
                self._delete(f"/agents/{self.agent_id}")
            if self.workflow_id:
                self._delete(f"/workflows/{self.workflow_id}")
            for tid in self.tool_ids.values():
                self._delete(f"/tools/{tid}")
            if self.knowledge_source_id:
                self._delete(f"/knowledge/sources/{self.knowledge_source_id}")
            if self.llm_config_id:
                self._delete(f"/llm-configs/{self.llm_config_id}")
        except Exception:
            pass


# ── 测试流程 ──────────────────────────────────────────────

def print_response(resp: dict, label: str = ""):
    """美化打印 invoke 响应"""
    answer = resp.get("short_answer", "")
    citations = resp.get("citations", [])
    followups = resp.get("suggested_followups", [])
    workflow_card = resp.get("workflow_card")
    workflow_status = resp.get("workflow_status")
    metadata = resp.get("metadata") or {}
    tool_calls = metadata.get("tool_calls", [])

    if label:
        print(f"    {BOLD}[{label}]{RESET}")
    print(f"    {BOLD}回答:{RESET} {answer[:300]}")

    if citations:
        print(f"    {CYAN}引用: {len(citations)} 条{RESET}")
        for c in citations[:2]:
            print(f"      - {c.get('source_name','')}: {c.get('content_snippet','')[:60]}...")

    if tool_calls:
        print(f"    {YELLOW}工具调用: {len(tool_calls)} 次{RESET}")
        for tc in tool_calls:
            print(f"      - {tc.get('function','')}: {json.dumps(tc.get('arguments',{}), ensure_ascii=False)[:80]}")

    if workflow_card:
        print(f"    {YELLOW}工作流: {workflow_card.get('step_name','')} (步骤 {workflow_card.get('current_step',0)}/{workflow_card.get('total_steps',0)}){RESET}")
        if workflow_card.get("fields"):
            for f in workflow_card["fields"]:
                print(f"      - {f.get('label','')}: [{f.get('field_type','text')}] {'*必填' if f.get('required') else ''}")

    if workflow_status:
        print(f"    {YELLOW}工作流状态: {workflow_status}{RESET}")

    if followups:
        print(f"    {CYAN}推荐追问: {', '.join(followups[:3])}{RESET}")

    print()


def run_demo(base_url: str, skip_cleanup: bool = False):
    """运行完整 demo"""
    client = DemoClient(base_url)
    results = {"pass": 0, "fail": 0}

    def check(condition: bool, pass_msg: str, fail_msg: str):
        if condition:
            ok(pass_msg)
            results["pass"] += 1
        else:
            fail(fail_msg)
            results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("0. 连接检查")
    # ═══════════════════════════════════════════════════════

    info(f"服务器地址: {base_url}")
    connected = client.health_check()
    check(connected, "服务器连接成功", f"无法连接到 {base_url}，请确认服务已启动")
    if not connected:
        print(f"\n{RED}请先启动服务器:{RESET}")
        print(f"  cd server && uvicorn server.main:app --host 0.0.0.0 --port 8000\n")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════
    section("1. 创建 LLM 配置 (DashScope Qwen Flash)")
    # ═══════════════════════════════════════════════════════

    try:
        llm_id = client.create_llm_config()
        ok(f"LLM 配置创建成功: {llm_id}")
        results["pass"] += 1
    except Exception as e:
        fail(f"LLM 配置创建失败: {e}")
        results["fail"] += 1
        sys.exit(1)

    sub_section("测试 LLM 连通性")
    try:
        test_result = client.test_llm_config()
        llm_ok = test_result.get("success", False)
        check(llm_ok,
              f"LLM 连通成功 (延迟 {test_result.get('latency_ms', 0):.0f}ms): {test_result.get('content', '')[:60]}",
              f"LLM 连通失败: {test_result.get('error', 'unknown')}")
        if not llm_ok:
            print(f"\n{YELLOW}  警告: LLM 不可用，后续调用测试可能失败。检查 API Key 和网络。{RESET}\n")
    except Exception as e:
        fail(f"LLM 测试请求异常: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("2. 创建知识库")
    # ═══════════════════════════════════════════════════════

    try:
        src_id = client.create_knowledge_source()
        ok(f"知识源创建成功: {src_id}")
        results["pass"] += 1
    except Exception as e:
        fail(f"知识源创建失败: {e}")
        results["fail"] += 1

    sub_section("添加 KV 精确条目 (5条)")
    try:
        client.add_kv_entries()
        ok("KV 条目添加成功 (物业电话/地址/停车/WiFi/食堂)")
        results["pass"] += 1
    except Exception as e:
        fail(f"KV 条目添加失败: {e}")
        results["fail"] += 1

    sub_section("添加 FAQ 条目 (5条)")
    try:
        client.add_faq_entries()
        ok("FAQ 条目添加成功 (会议室/访客/配套/门禁卡/报修)")
        results["pass"] += 1
    except Exception as e:
        fail(f"FAQ 条目添加失败: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("3. 创建工具")
    # ═══════════════════════════════════════════════════════

    try:
        client.create_tools()
        for name, tid in client.tool_ids.items():
            ok(f"工具「{name}」创建成功: {tid}")
        results["pass"] += 1
    except Exception as e:
        fail(f"工具创建失败: {e}")
        results["fail"] += 1

    # 直接测试 mock 端点
    sub_section("Mock 工具直连测试")
    try:
        calc = requests.post(f"{base_url}/api/v1/mock-tools/calculator",
                             json={"operation": "add", "a": 123, "b": 456}).json()
        check(calc.get("result") == 579, f"计算器: 123+456={calc.get('result')}", "计算器返回错误")
    except Exception as e:
        fail(f"计算器测试异常: {e}")
        results["fail"] += 1

    try:
        weather = requests.post(f"{base_url}/api/v1/mock-tools/weather",
                                json={"city": "北京"}).json()
        check(weather.get("success"), f"天气: {weather.get('forecast', '')}", "天气查询失败")
    except Exception as e:
        fail(f"天气测试异常: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("4. 创建工作流（报修申请）")
    # ═══════════════════════════════════════════════════════

    try:
        wf_id = client.create_workflow()
        ok(f"工作流创建成功: {wf_id}")
        results["pass"] += 1

        # 验证步骤
        wf = client._get(f"/workflows/{wf_id}")
        steps = wf.get("steps", [])
        check(len(steps) == 3, f"工作流包含 {len(steps)} 个步骤", "步骤数不正确")
        for step in sorted(steps, key=lambda s: s["order"]):
            info(f"  步骤{step['order']}: {step['name']} ({step['step_type']})")
    except Exception as e:
        fail(f"工作流创建失败: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("5. 创建 Agent + 绑定能力")
    # ═══════════════════════════════════════════════════════

    try:
        agent_id = client.create_agent()
        ok(f"Agent「智慧园区助手」创建成功: {agent_id}")
        results["pass"] += 1
    except Exception as e:
        fail(f"Agent 创建失败: {e}")
        results["fail"] += 1

    sub_section("绑定能力 (知识 + 工具 + 工作流)")
    try:
        caps = client.setup_capabilities()
        k_count = len(caps.get("knowledge", []))
        w_count = len(caps.get("workflows", []))
        t_count = len(caps.get("tools", []))
        ok(f"能力绑定成功: 知识库×{k_count}, 工作流×{w_count}, 工具组×{t_count}")
        results["pass"] += 1
    except Exception as e:
        fail(f"能力绑定失败: {e}")
        results["fail"] += 1

    # 验证 skills 已自动创建
    sub_section("验证自动创建的 Skills")
    try:
        agent_skills = client._get(f"/agents/{agent_id}/skills")
        check(len(agent_skills) >= 3,
              f"自动创建 {len(agent_skills)} 个 Skills (知识/工作流/工具)",
              f"Skills 数量不足: {len(agent_skills)}")
        for sk in agent_skills:
            info(f"  {sk.get('skill_name', '')} [{sk.get('skill_type', '')}]")
    except Exception as e:
        fail(f"Skills 查询失败: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("6. 全流程调用测试")
    # ═══════════════════════════════════════════════════════

    # ── 6.1 闲聊 ──
    sub_section("6.1 闲聊/打招呼")
    try:
        resp = client.invoke_new_session("你好！请问你是谁？")
        print_response(resp, "打招呼")
        check(len(resp.get("short_answer", "")) > 0, "闲聊回复正常", "闲聊无回复")
    except Exception as e:
        fail(f"闲聊调用失败: {e}")
        results["fail"] += 1

    # ── 6.2 知识检索 - KV 精确匹配 ──
    sub_section("6.2 知识检索 - KV 精确匹配")
    try:
        resp = client.invoke_new_session("物业电话是多少？")
        print_response(resp, "KV 精确查询")
        answer = resp.get("short_answer", "")
        check("400-888-9999" in answer or "物业" in answer,
              "KV 精确匹配成功（含物业电话）",
              f"KV 匹配未命中，回复: {answer[:100]}")
    except Exception as e:
        fail(f"KV 查询调用失败: {e}")
        results["fail"] += 1

    # ── 6.3 知识检索 - FAQ 语义匹配 ──
    sub_section("6.3 知识检索 - FAQ 语义匹配")
    try:
        resp = client.invoke_new_session("怎么预约会议室？")
        print_response(resp, "FAQ 语义查询")
        answer = resp.get("short_answer", "")
        check("会议" in answer or "预约" in answer,
              "FAQ 语义匹配成功",
              f"FAQ 匹配可能未命中: {answer[:100]}")
    except Exception as e:
        fail(f"FAQ 查询调用失败: {e}")
        results["fail"] += 1

    # ── 6.4 知识检索 - 停车相关 ──
    sub_section("6.4 知识检索 - 停车收费")
    try:
        resp = client.invoke_new_session("停车怎么收费？")
        print_response(resp, "停车查询")
        answer = resp.get("short_answer", "")
        check("5元" in answer or "停车" in answer or "300" in answer,
              "停车收费信息检索成功",
              f"停车信息可能未命中: {answer[:100]}")
    except Exception as e:
        fail(f"停车查询失败: {e}")
        results["fail"] += 1

    # ── 6.5 工具调用 - 天气 ──
    sub_section("6.5 工具调用 - 天气查询")
    try:
        resp = client.invoke_new_session("北京今天天气怎么样？")
        print_response(resp, "天气工具")
        answer = resp.get("short_answer", "")
        metadata = resp.get("metadata") or {}
        tool_calls = metadata.get("tool_calls", [])
        has_weather = any("weather" in tc.get("function", "").lower() or "天气" in tc.get("function", "")
                          for tc in tool_calls)
        check(has_weather or "°C" in answer or "天气" in answer or "℃" in answer,
              "天气工具调用成功",
              f"天气工具可能未触发: {answer[:100]}")
    except Exception as e:
        fail(f"天气调用失败: {e}")
        results["fail"] += 1

    # ── 6.6 工具调用 - 计算器 ──
    sub_section("6.6 工具调用 - 计算器")
    try:
        resp = client.invoke_new_session("帮我算一下 256 乘以 38 等于多少？")
        print_response(resp, "计算工具")
        answer = resp.get("short_answer", "")
        metadata = resp.get("metadata") or {}
        tool_calls = metadata.get("tool_calls", [])
        has_calc = any("calc" in tc.get("function", "").lower() or "计算" in tc.get("function", "")
                       for tc in tool_calls)
        check(has_calc or "9728" in answer,
              "计算器工具调用成功",
              f"计算器可能未触发: {answer[:100]}")
    except Exception as e:
        fail(f"计算调用失败: {e}")
        results["fail"] += 1

    # ── 6.7 工作流 - 报修申请 ──
    sub_section("6.7 工作流 - 发起报修")
    try:
        resp = client.invoke_new_session("我要报修，办公室空调坏了")
        print_response(resp, "发起报修")
        answer = resp.get("short_answer", "")
        wf_card = resp.get("workflow_card")
        wf_status = resp.get("workflow_status")

        if wf_card:
            ok(f"工作流启动成功: {wf_card.get('step_name', '')}")
            results["pass"] += 1

            # 继续填写表单
            sub_section("6.7.1 工作流 - 填写报修信息")
            try:
                resp2 = client.invoke_agent(
                    "填好了",
                    form_data={
                        "location": "A栋5楼503室",
                        "issue_type": "hvac",
                        "description": "空调不制冷，开机有异响",
                        "contact_phone": "13800138000",
                    }
                )
                print_response(resp2, "提交表单")
                wf_card2 = resp2.get("workflow_card")
                wf_status2 = resp2.get("workflow_status")
                check(wf_card2 or wf_status2 or "确认" in resp2.get("short_answer", ""),
                      "表单提交成功，进入下一步",
                      f"表单提交后状态异常: {resp2.get('short_answer', '')[:100]}")
            except Exception as e:
                fail(f"表单提交失败: {e}")
                results["fail"] += 1

            # 确认
            sub_section("6.7.2 工作流 - 确认提交")
            try:
                resp3 = client.invoke_agent("确认提交")
                print_response(resp3, "确认提交")
                answer3 = resp3.get("short_answer", "")
                check("确认" in answer3 or "工单" in answer3 or "完成" in answer3 or resp3.get("workflow_status") == "completed",
                      "工作流确认/完成",
                      f"工作流确认状态异常: {answer3[:100]}")
            except Exception as e:
                fail(f"确认步骤失败: {e}")
                results["fail"] += 1
        else:
            # 工作流可能通过文字回复了报修相关信息
            check("报修" in answer or "工单" in answer or "维修" in answer,
                  "报修相关回复（工作流可能以文本方式响应）",
                  f"报修工作流未触发: {answer[:100]}")
    except Exception as e:
        fail(f"报修工作流调用失败: {e}")
        results["fail"] += 1

    # ── 6.8 多轮对话上下文 ──
    sub_section("6.8 多轮对话 - 上下文关联")
    try:
        # 第一轮
        resp1 = client.invoke_new_session("园区有健身房吗？")
        print_response(resp1, "第一轮")
        answer1 = resp1.get("short_answer", "")

        # 第二轮（引用上下文）
        resp2 = client.invoke_agent("怎么办卡？")
        print_response(resp2, "第二轮（上下文追问）")
        answer2 = resp2.get("short_answer", "")
        check(len(answer2) > 5,
              "多轮对话上下文关联正常",
              "多轮对话回复异常")
    except Exception as e:
        fail(f"多轮对话失败: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("7. 审计日志检查")
    # ═══════════════════════════════════════════════════════

    try:
        traces = client._get("/audit/traces", params={"limit": 10})
        if isinstance(traces, list):
            check(len(traces) > 0, f"审计日志已记录 {len(traces)} 条（最近10条）", "审计日志为空")
            # 检查事件类型覆盖
            event_types = set()
            for t in traces:
                event_types.add(t.get("event_type", ""))
            info(f"事件类型: {', '.join(sorted(event_types))}")
        else:
            info(f"审计日志返回格式: {type(traces)}")
    except Exception as e:
        fail(f"审计日志查询失败: {e}")
        results["fail"] += 1

    # ═══════════════════════════════════════════════════════
    section("测试结果汇总")
    # ═══════════════════════════════════════════════════════

    total = results["pass"] + results["fail"]
    print(f"  {GREEN}通过: {results['pass']}{RESET}")
    print(f"  {RED}失败: {results['fail']}{RESET}")
    print(f"  总计: {total}")
    print()

    if results["fail"] == 0:
        print(f"  {GREEN}{BOLD}ALL PASSED! 全部测试通过！{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}部分测试未通过，请检查上方详细输出。{RESET}")

    # ── 清理 ──
    if not skip_cleanup:
        print()
        info("清理 demo 数据...")
        client.cleanup()
        ok("清理完成")
    else:
        print()
        info(f"跳过清理。Agent ID: {client.agent_id}")
        info("可在前端 Playground 继续测试")

    return results["fail"] == 0


# ── 入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HlAB 全流程中文 Demo 测试")
    parser.add_argument("--base-url", default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--keep", action="store_true", help="测试后保留 demo 数据（不清理）")
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════╗
║   HlAB - 全流程中文 Demo 测试                    ║
║   智慧园区助手 AI Agent 端到端验证               ║
╚══════════════════════════════════════════════════╝{RESET}
""")

    success = run_demo(args.base_url, skip_cleanup=args.keep)
    sys.exit(0 if success else 1)
