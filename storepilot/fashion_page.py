"""The owner's conversation, evidence notebook and final decisions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from .fashion_agent import (
    DEFAULT_MODEL,
    MARKETS,
    FashionJournal,
    GeminiResearchModel,
    journal_path,
    research_turn,
)
from .fashion_sources import FashionSources, ResearchError

ZH = {
    "Fashion scout": "时尚侦察员",
    "Talk through what to stock next. Check the evidence, then make the buying decision yourself.": "聊聊下一步选什么款。先看证据，再由你决定是否进货。",
    "Example walkthrough": "体验示例流程",
    "Live research": "实时调研",
    "Research mode": "调研方式",
    "Your store": "你的门店",
    "Store market": "门店所在市场",
    "Choose a country": "请选择国家",
    "Town or city (optional)": "城市（可选）",
    "Who shops with you?": "你的主要顾客是谁？",
    "Clothing you sell": "你经营哪类服装？",
    "For example: students and young professionals who want everyday clothes": "例如：喜欢日常穿搭的学生与年轻上班族",
    "For example: casual womenswear, denim and light jackets": "例如：女装休闲款、牛仔裤和薄外套",
    "Typical minimum price": "常见最低售价",
    "Typical maximum price": "常见最高售价",
    "Search language": "检索语言",
    "Connect research services": "连接调研服务",
    "Gemini API key": "Gemini API 密钥",
    "SerpApi data key": "SerpApi 数据密钥",
    "Gemini model": "Gemini 模型",
    "Keys stay in this session or your server environment; they are not saved in the research journal.": "密钥仅保存在本次会话或服务器环境中，不会写入调研记录。",
    "Live questions send your question, store profile and recent conversation/evidence to Gemini, and search phrases to SerpApi. Each turn allows at most 6 data requests and 7 model requests; provider charges may apply.": "实时提问会将问题、门店信息与近期对话及证据发送给 Gemini，并将检索词发送给 SerpApi。每轮最多 6 次数据请求和 7 次模型请求，服务商可能收费。",
    "Google Trends and retailer listings come through SerpApi. Instagram checks cover public search-index excerpts only. This is not direct access to Instagram engagement or Amazon sales data.": "Google 搜索趋势与零售商品信息通过 SerpApi 获取。Instagram 检索仅覆盖公开搜索摘要，不能直接读取 Instagram 互动统计或亚马逊销量。",
    "Choose your country and connect both keys to start a live conversation.": "选择门店所在国家并填写两项密钥，即可开始实时对话。",
    "This is a scripted example with invented evidence, not current fashion advice. Live research uses your connected services.": "这是使用虚构证据的固定示例，不代表当前流行趋势。实时调研会使用你连接的服务。",
    "Show an example conversation": "查看示例对话",
    "What would you like to find out?": "你想了解什么？",
    "Looking into your question…": "正在调研你的问题…",
    "Searching recent sources": "检索近期来源",
    "Checking search interest": "查看搜索兴趣",
    "Checking retailer prices": "核对零售价格",
    "Research complete": "调研完成",
    "Research could not finish": "调研未完成",
    "No final answer was saved. {0}": "未保存最终答复。{0}",
    "Observation": "观察结果",
    "Interpretation": "分析判断",
    "Suggestion": "行动建议",
    "Evidence and working notes": "证据与工作记录",
    "Source": "来源",
    "Open source": "打开来源",
    "Collected": "采集时间",
    "Published label": "来源标注日期",
    "Query": "检索词",
    "Price": "价格",
    "Retailer": "零售商",
    "Budget match": "价格是否符合预算",
    "Yes": "是",
    "No": "否",
    "Unknown currency": "币种未核实",
    "Search interest is normalized within each series. It does not measure purchases, customer approval or when a trend will arrive locally.": "搜索兴趣在每条序列内归一化，不代表购买量、顾客喜爱程度或趋势何时传到本地。",
    "Four-week change: {0:+.1f}%": "最近四周相比前四周：{0:+.1f}%",
    "There is not enough regular, recent search data to report a reliable change.": "近期连续搜索数据不足，暂不报告变化幅度。",
    "Research actions": "调研操作记录",
    "Download research record": "下载调研记录",
    "Your decision": "你的决定",
    "Item or style": "商品或款式",
    "Keep watching": "继续观察",
    "Try a small batch": "考虑小批量试卖",
    "Not for my customers": "不适合我的顾客",
    "Why does this fit—or not fit—your customers?": "为什么它适合或不适合你的顾客？",
    "Save my decision": "保存我的决定",
    "Decision saved": "决定已保存",
    "This records your judgement only. It does not order stock or change the replenishment plan.": "这里只记录你的判断，不会自动订货或修改补货计划。",
    "Earlier decisions": "之前的决定",
    "Decision": "决定",
    "Note": "备注",
    "Date": "日期",
    "Germany": "德国",
    "France": "法国",
    "Spain": "西班牙",
    "Italy": "意大利",
    "Netherlands": "荷兰",
    "Belgium": "比利时",
    "Ireland": "爱尔兰",
    "Portugal": "葡萄牙",
    "Austria": "奥地利",
    "Finland": "芬兰",
    "United Kingdom": "英国",
    "Switzerland": "瑞士",
    "Poland": "波兰",
    "Example brief": "示例简报",
    "Synthetic search series": "虚构搜索序列",
    "Fictional everyday-clothing shop": "虚构日常服装店",
    "What affordable everyday style could I test with a small batch?": "有什么平价日常款适合小批量试卖？",
    "Let's use wide-leg jeans to demonstrate how we would check an idea. These numbers are invented for the walkthrough.": "我们用阔腿牛仔裤演示如何核实一个想法。以下数字均为示例而虚构。",
    "In this example, recent search interest rose, and one fictional shop lists a pair at €39.": "在这个示例中，近期搜索兴趣上升，一家虚构商店标价为 39 欧元。",
    "That makes the style worth investigating, but one listing and search growth cannot establish what your customers will buy.": "这让该款式值得继续了解，但单个商品页面和搜索增长无法证明你的顾客会购买。",
    "Ask regular customers about fit and price, then consider a small trial if their answers support it.": "先询问常客对版型和价格的看法；如果反馈合适，再考虑少量试卖。",
    "Who usually buys jeans from you, and what fit do they prefer?": "通常哪些顾客在你这里买牛仔裤？他们喜欢什么版型？",
}


def demo_turn(profile, language):
    def t(text):
        return ZH.get(text, text) if language == "zh" else text

    return {
        "id": "fashion-walkthrough-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "mode": "demo",
        "model": "scripted-example",
        "data_calls": 0,
        "language": language,
        "question": t("What affordable everyday style could I test with a small batch?"),
        "answer": {
            "reply": t(
                "Let's use wide-leg jeans to demonstrate how we would check an idea. These numbers are invented for the walkthrough."
            ),
            "claims": [
                {
                    "kind": "observation",
                    "text": t(
                        "In this example, recent search interest rose, and one fictional shop lists a pair at €39."
                    ),
                    "evidence_ids": ["DEMO-E1", "DEMO-E2"],
                },
                {
                    "kind": "interpretation",
                    "text": t(
                        "That makes the style worth investigating, but one listing and search growth cannot establish what your customers will buy."
                    ),
                    "evidence_ids": ["DEMO-E1", "DEMO-E2"],
                },
                {
                    "kind": "suggestion",
                    "text": t(
                        "Ask regular customers about fit and price, then consider a small trial if their answers support it."
                    ),
                    "evidence_ids": [],
                },
            ],
            "question": t("Who usually buys jeans from you, and what fit do they prefer?"),
        },
        "evidence": {
            "DEMO-E1": {
                "kind": "sample",
                "title": t("Synthetic search series"),
                "url": "",
                "excerpt": "Invented weekly values: 20, 22, 21, 23, 30, 35, 38, 40. Not observed data.",
            },
            "DEMO-E2": {
                "kind": "sample",
                "title": t("Fictional everyday-clothing shop"),
                "url": "",
                "excerpt": "Invented retail listing: wide-leg jeans, EUR 39. Not a real offer.",
            },
        },
        "trace": [],
    }


def render_fashion_scout(language):
    def t(text, *args):
        text = ZH.get(text, text) if language == "zh" else text
        return text.format(*args) if args else text

    if st.session_state.get("fashion_label_language") != language:
        for key in list(st.session_state):
            if key in {"fashion_country", "fashion_mode"} or key.startswith("fashion_decision:"):
                st.session_state[key] = st.session_state[key]
        st.session_state["fashion_label_language"] = language
    st.header(t("Fashion scout"))
    st.write(
        t(
            "Talk through what to stock next. Check the evidence, then make the buying decision yourself."
        )
    )
    st.session_state.setdefault("fashion_mode", "demo")
    mode = st.radio(
        t("Research mode"),
        ["demo", "live"],
        key="fashion_mode",
        horizontal=True,
        format_func={"demo": t("Example walkthrough"), "live": t("Live research")}.get,
    )
    draft = st.session_state.get("fashion_profile_draft", {})
    with st.expander(t("Your store"), expanded=mode == "live"):
        st.session_state.setdefault("fashion_country", draft.get("country", ""))
        country = st.selectbox(
            t("Store market"),
            ["", *MARKETS],
            key="fashion_country",
            format_func={"": t("Choose a country"), **{k: t(v[0]) for k, v in MARKETS.items()}}.get,
        )
        currency, search_lang = MARKETS.get(country, ("", "EUR", "en"))[1:]
        location = st.text_input(
            t("Town or city (optional)"),
            value=draft.get("location", ""),
            max_chars=800,
            key="fashion_location",
        )
        customers = st.text_input(
            t("Who shops with you?"),
            value=draft.get("customers", ""),
            placeholder=t(
                "For example: students and young professionals who want everyday clothes"
            ),
            max_chars=800,
            key="fashion_customers",
        )
        focus = st.text_input(
            t("Clothing you sell"),
            value=draft.get("focus", ""),
            placeholder=t("For example: casual womenswear, denim and light jackets"),
            max_chars=800,
            key="fashion_focus",
        )
        left, right = st.columns(2)
        low = left.number_input(
            t("Typical minimum price") + f" ({currency})",
            0.0,
            9999.0,
            float(draft.get("price_min", 20)),
            key="fashion_price_min",
        )
        high = right.number_input(
            t("Typical maximum price") + f" ({currency})",
            1.0,
            10000.0,
            float(draft.get("price_max", 80)),
            key="fashion_price_max",
        )
        profile = {
            "country": country,
            "currency": currency,
            "search_language": search_lang,
            "location": location,
            "customers": customers,
            "focus": focus,
            "price_min": low,
            "price_max": high,
        }
        st.session_state["fashion_profile_draft"] = profile
    gemini_key = serp_key = ""
    model_name = DEFAULT_MODEL
    if mode == "live":
        with st.expander(t("Connect research services"), expanded=True):
            gemini_key = st.text_input(
                t("Gemini API key"), type="password", key="fashion_gemini_key"
            ) or os.environ.get("GEMINI_API_KEY", "")
            serp_key = st.text_input(
                t("SerpApi data key"), type="password", key="fashion_serpapi_key"
            ) or os.environ.get("SERPAPI_API_KEY", "")
            model_name = st.text_input(
                t("Gemini model"),
                value=os.environ.get("STOREPILOT_GEMINI_MODEL", DEFAULT_MODEL),
                key="fashion_model",
            )
            st.caption(
                t(
                    "Keys stay in this session or your server environment; they are not saved in the research journal."
                )
            )
            st.markdown(
                "[Google AI Studio](https://aistudio.google.com/apikey) · [SerpApi](https://serpapi.com/manage-api-key)"
            )
        st.caption(
            t(
                "Live questions send your question, store profile and recent conversation/evidence to Gemini, and search phrases to SerpApi. Each turn allows at most 6 data requests and 7 model requests; provider charges may apply."
            )
        )
        st.info(
            t(
                "Google Trends and retailer listings come through SerpApi. Instagram checks cover public search-index excerpts only. This is not direct access to Instagram engagement or Amazon sales data."
            )
        )
    else:
        st.info(
            t(
                "This is a scripted example with invented evidence, not current fashion advice. Live research uses your connected services."
            )
        )

    database = os.environ.get("STOREPILOT_DB", "storepilot.db")
    journal = FashionJournal(journal_path(database, demo=mode == "demo"))
    if mode == "demo":
        # Fixed demo context and isolated journal: never mix invented evidence into live chat.
        profile = {
            "country": "DE",
            "currency": "EUR",
            "search_language": "de",
            "location": "Example only",
            "customers": "Example only",
            "focus": "Example only",
            "price_min": 20.0,
            "price_max": 80.0,
        }
        if st.button(t("Show an example conversation"), key="fashion_example"):
            st.session_state["fashion_demo_turn"] = demo_turn(profile, language)
            journal.save(st.session_state["fashion_demo_turn"])
        history = (
            [st.session_state["fashion_demo_turn"]]
            if "fashion_demo_turn" in st.session_state
            else []
        )
    else:
        history = journal.history(profile)
    ready = mode == "live" and bool(country and gemini_key and serp_key)
    if mode == "live" and not ready:
        st.caption(t("Choose your country and connect both keys to start a live conversation."))

    def show_turn(turn):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"]["reply"])
            for claim in turn["answer"]["claims"]:
                st.caption(
                    t(
                        {
                            "observation": "Observation",
                            "interpretation": "Interpretation",
                            "suggestion": "Suggestion",
                        }[claim["kind"]]
                    )
                )
                st.write(claim["text"])
                for ref in claim["evidence_ids"]:
                    item = turn["evidence"][ref]
                    if item.get("url"):
                        st.link_button(
                            f"{ref.rsplit('-', 1)[-1]} · {item['title'][:85]}", item["url"]
                        )
                    else:
                        st.caption(f"{ref} · {item['title']}")
            if turn["answer"]["question"]:
                st.write(turn["answer"]["question"])
            with st.expander(t("Evidence and working notes")):
                st.caption(turn["created_at"])
                for ref, item in turn["evidence"].items():
                    st.write(f"{ref.rsplit('-', 1)[-1]} · {item['title']}")
                    if item.get("excerpt"):
                        st.text(item["excerpt"])
                    if item.get("collected_at"):
                        st.caption(f"{t('Collected')}: {item['collected_at']}")
                    if item.get("published_label"):
                        st.caption(f"{t('Published label')}: {item['published_label']}")
                    if item["kind"] == "search_interest":
                        st.caption(
                            t(
                                "Search interest is normalized within each series. It does not measure purchases, customer approval or when a trend will arrive locally."
                            )
                        )
                        if item.get("change_pct") is not None:
                            st.write(t("Four-week change: {0:+.1f}%", item["change_pct"]))
                        else:
                            st.write(
                                t(
                                    "There is not enough regular, recent search data to report a reliable change."
                                )
                            )
                        if item.get("points"):
                            st.line_chart(pd.DataFrame(item["points"]).set_index("date"))
                    if item["kind"] == "retailer_listing":
                        st.write(
                            f"{t('Retailer')}: {item['retailer']} · {t('Price')}: {item['price_label']}"
                        )
                        match = (
                            "Unknown currency"
                            if item["within_budget"] is None
                            else "Yes"
                            if item["within_budget"]
                            else "No"
                        )
                        st.caption(f"{t('Budget match')}: {t(match)}")
                    if item.get("url"):
                        st.link_button(t("Open source"), item["url"])
                if turn["trace"]:
                    st.caption(t("Research actions"))
                    st.dataframe(pd.DataFrame(turn["trace"]), hide_index=True, width="stretch")
                st.download_button(
                    t("Download research record"),
                    json.dumps(turn, ensure_ascii=False, indent=2),
                    f"fashion-research-{turn['id']}.json",
                    "application/json",
                    key=f"fashion_export:{turn['id']}",
                )

    for turn in history:
        show_turn(turn)
    question = st.chat_input(
        t("What would you like to find out?"),
        disabled=not ready,
        max_chars=1800,
        key="fashion_chat",
    )
    if question:
        with st.status(t("Looking into your question…"), expanded=True) as status:

            def progress(tool, query):
                labels = {
                    "search_fashion": "Searching recent sources",
                    "check_interest": "Checking search interest",
                    "check_prices": "Checking retailer prices",
                }
                st.write(f"{t(labels[tool])}: {query}")

            try:
                turn = research_turn(
                    question,
                    profile,
                    history,
                    GeminiResearchModel(gemini_key, model_name),
                    FashionSources(serp_key),
                    language=language,
                    on_progress=progress,
                    owner_decisions=journal.decisions(profile),
                )
                journal.save(turn)
            except ResearchError as exc:
                status.update(label=t("Research could not finish"), state="error")
                st.error(t("No final answer was saved. {0}", str(exc)))
            else:
                status.update(label=t("Research complete"), state="complete")
                st.rerun()
    if history:
        latest = history[-1]
        st.subheader(t("Your decision"))
        st.caption(
            t(
                "This records your judgement only. It does not order stock or change the replenishment plan."
            )
        )
        with st.container(border=True):
            item = st.text_input(
                t("Item or style"), max_chars=150, key=f"fashion_item:{latest['id']}"
            )
            decision = st.radio(
                t("Decision"),
                ["watch", "trial", "pass"],
                horizontal=True,
                key=f"fashion_decision:{latest['id']}",
                format_func={
                    "watch": t("Keep watching"),
                    "trial": t("Try a small batch"),
                    "pass": t("Not for my customers"),
                }.get,
            )
            note = st.text_area(
                t("Why does this fit—or not fit—your customers?"),
                max_chars=2000,
                key=f"fashion_note:{latest['id']}",
            )
            save = st.button(t("Save my decision"), key=f"fashion_save:{latest['id']}")
        if save:
            try:
                journal.decide(latest, item, decision, note)
                st.success(t("Decision saved"))
            except ResearchError as exc:
                st.error(str(exc))
        decisions = journal.decisions(profile)
        if decisions:
            with st.expander(t("Earlier decisions")):
                frame = pd.DataFrame(decisions)[["item", "decision", "note", "created_at"]]
                frame["decision"] = frame.decision.map(
                    {
                        "watch": t("Keep watching"),
                        "trial": t("Try a small batch"),
                        "pass": t("Not for my customers"),
                    }
                )
                st.dataframe(
                    frame.rename(
                        columns={
                            "item": t("Item or style"),
                            "decision": t("Decision"),
                            "note": t("Note"),
                            "created_at": t("Date"),
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
