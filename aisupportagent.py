#!/usr/bin/env python3
"""
AI Ops Support Agent - Complete Single File Implementation
Microsoft Foundry · Azure OpenAI · Full Guardrails

Usage:
    1. Set environment variables:
       export AOAI_ENDPOINT="https://your-resource.openai.azure.com/"
       export AOAI_API_KEY="your-api-key"
       export AOAI_DEPLOYMENT="gpt-4-deployment"
    
    2. Run: python ai_ops_agent.py
    
    3. Open browser to http://localhost:8501
"""

import os
import sys
import re
import json
import logging
import time
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

import requests

# Try to import required packages, install if missing
try:
    import streamlit as st
    import openai
    from openai import AzureOpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                          "streamlit", "openai", "python-dotenv", "requests"])
    print("Packages installed. Please run the script again.")
    sys.exit(0)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AzureOpenAIConfig:
    """Azure OpenAI configuration"""
    endpoint: str = os.getenv("AOAI_ENDPOINT", "")
    api_key: str = os.getenv("AOAI_API_KEY", "")
    deployment: str = os.getenv("AOAI_DEPLOYMENT", "")
    api_version: str = "2024-02-15-preview"
    
    def validate(self) -> bool:
        return all([self.endpoint, self.api_key, self.deployment])


@dataclass
class AgentConfig:
    """Agent configuration with safe defaults"""
    temperature: float = float(os.getenv("DEFAULT_TEMPERATURE", "1.0"))
    max_tokens: int = int(os.getenv("DEFAULT_MAX_TOKENS", "350"))
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: Optional[int] = 42
    
    # Guardrail settings
    enable_content_filtering: bool = True
    enable_pii_detection: bool = True
    enable_jailbreak_detection: bool = True
    
    # System prompt
    system_prompt: str = """You are a specialized AI support agent for ServiceNow ticket summarization.

Your responsibilities:
1. Summarize support tickets in exactly 3 concise sentences
2. ALWAYS state explicitly whether the ticket is at risk of breaching its SLA
3. Reference the SLA policy thresholds provided in context
4. If information is insufficient to judge SLA risk, say so rather than guessing

Guidelines:
- Be factual and precise
- Do not provide financial, legal, or HR advice
- Stay within ticket summarization scope
- If asked about topics outside your scope, politely decline

SLA Policy Thresholds (for reference):
- P1 (Critical): Response within 1 hour, Resolution within 4 hours
- P2 (High): Response within 2 hours, Resolution within 8 hours
- P3 (Medium): Response within 4 hours, Resolution within 24 hours
- P4 (Low): Response within 8 hours, Resolution within 48 hours
"""


# ============================================================================
# GUARDRAILS
# ============================================================================

class ContentSafetyGuardrails:
    """Content safety and policy guardrails"""
    
    JAILBREAK_PATTERNS = [
        r"ignore previous instructions",
        r"forget your guidelines",
        r"you are now (an AI|a model) without restrictions",
        r"pretend to be (unrestricted|unfiltered)",
        r"system prompt",
        r"developer prompt",
        r"role-play as (evil|unethical|harmful)",
        r"bypass safety",
        r"override content policy",
        r"(hack|break|bypass) your (safety|security|protocol)",
        r"new (mission|objective|goal)",
        r"disregard (all|any) (previous|prior) (instructions|rules)",
    ]
    
    PII_PATTERNS = {
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
    }
    
    PROHIBITED_TOPICS = [
        "financial advice",
        "legal advice",
        "medical advice",
        "investment recommendations",
        "trading strategies",
        "tax advice",
    ]
    
    def __init__(self, enable_pii: bool = True):
        self.enable_pii = enable_pii
        self.jailbreak_regex = [re.compile(p, re.IGNORECASE) for p in self.JAILBREAK_PATTERNS]
        self.pii_regex = {n: re.compile(p) for n, p in self.PII_PATTERNS.items()}
    
    def check_jailbreak(self, text: str) -> Tuple[bool, List[str]]:
        violations = []
        for pattern in self.jailbreak_regex:
            if pattern.search(text):
                violations.append(f"Potential jailbreak: {pattern.pattern}")
        return len(violations) == 0, violations
    
    def check_pii(self, text: str) -> Tuple[bool, Dict[str, List[str]]]:
        if not self.enable_pii:
            return True, {}
        found = {}
        for name, pattern in self.pii_regex.items():
            matches = pattern.findall(text)
            if matches:
                found[name] = matches
        return len(found) == 0, found
    
    def filter_content(self, text: str) -> Dict[str, Any]:
        violations = []
        severity = 0.0
        filtered = text
        
        # Check jailbreak
        safe, jb_violations = self.check_jailbreak(text)
        if not safe:
            violations.extend(jb_violations)
            severity += 0.7
        
        # Check PII
        safe, pii_found = self.check_pii(text)
        if not safe and self.enable_pii:
            for name, matches in pii_found.items():
                for match in matches:
                    filtered = filtered.replace(match, f"[REDACTED_{name.upper()}]")
                violations.append(f"PII detected: {name}")
            severity += 0.4
        
        # Check prohibited topics
        text_lower = text.lower()
        for topic in self.PROHIBITED_TOPICS:
            if topic in text_lower:
                violations.append(f"Prohibited topic: {topic}")
                severity += 0.6
        
        is_safe = len(violations) == 0
        if severity > 0.8:
            filtered = "[Content blocked due to policy violation]"
        
        return {
            "is_safe": is_safe,
            "violations": violations,
            "severity": min(severity, 1.0),
            "filtered_response": filtered
        }


class ExecutionGuardrails:
    """Execution parameter guardrails"""
    
    @classmethod
    def validate_parameters(cls, params: Dict) -> Dict:
        validated = params.copy()
        ranges = {
            "temperature": (0.0, 1.0),
            "max_tokens": (50, 500),
            "top_p": (0.5, 1.0),
        }
        for param, (min_val, max_val) in ranges.items():
            if param in validated:
                validated[param] = max(min_val, min(max_val, validated[param]))
        return validated
    
    @classmethod
    def get_defaults(cls) -> Dict:
        return {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 350,
        }


# ============================================================================
# AGENT
# ============================================================================

class FoundryAgent:
    """Microsoft Foundry AI Agent"""
    
    def __init__(self, azure_config: AzureOpenAIConfig, agent_config: AgentConfig):
        self.azure_config = azure_config
        self.agent_config = agent_config
        self.guardrails = ContentSafetyGuardrails(agent_config.enable_pii_detection)
        self.use_foundry_project = self._is_foundry_project_endpoint(self.azure_config.endpoint)
        self.client = self._init_client()
        self.conversation_history = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.requests_count = 0

    @staticmethod
    def _is_foundry_project_endpoint(endpoint: str) -> bool:
        endpoint_lower = (endpoint or "").lower()
        return "/api/projects/" in endpoint_lower or ".services.ai.azure.com" in endpoint_lower

    def _init_client(self):
        if self.use_foundry_project:
            return None
        try:
            return AzureOpenAI(
                azure_endpoint=self.azure_config.endpoint,
                api_key=self.azure_config.api_key,
                api_version=self.azure_config.api_version,
            )
        except Exception as e:
            logging.error(f"Failed to initialize client: {e}")
            raise

    def _call_foundry_project(self, messages: List[Dict], exec_params: Dict) -> Dict[str, Any]:
        api_key = self.azure_config.api_key
        if not api_key:
            raise ValueError("AOAI_API_KEY is required for Foundry project mode.")

        model_name = (self.azure_config.deployment or "").strip()
        if not model_name:
            raise ValueError("AOAI_DEPLOYMENT is required for Foundry project mode.")

        endpoint_root = self.azure_config.endpoint.rstrip("/")
        service_root = endpoint_root.split("/api/projects/")[0].rstrip("/") if "/api/projects/" in endpoint_root.lower() else endpoint_root
        model_url_variants = []

        base_variants = [
            service_root,
            service_root + "/openai",
            service_root + "/models",
            endpoint_root,
        ]
        for base in dict.fromkeys(base_variants):
            model_url_variants.extend([
                f"{base}/openai/deployments/{model_name}/chat/completions?api-version=2024-10-21",
                f"{base}/openai/deployments/{model_name}/chat/completions?api-version=2025-04-01-preview",
                f"{base}/models/chat/completions?api-version=2024-05-01-preview",
                f"{base}/models/chat/completions?api-version=2025-04-01-preview",
                f"{base}/openai/v1/chat/completions",
                f"{base}/chat/completions?api-version=2024-05-01-preview",
            ])

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": 1.0,
            "max_completion_tokens": exec_params.get("max_tokens", 350),
        }
        headers = {
            "api-key": api_key,
            "Content-Type": "application/json",
        }

        last_error = None
        for url in model_url_variants:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                    continue
                return response.json()
            except Exception as exc:
                last_error = str(exc)
                continue

        raise RuntimeError(last_error or "Failed to call Foundry project model endpoint.")
    
    def _prepare_messages(self, user_input: str, context: Optional[str] = None) -> List[Dict]:
        messages = [{"role": "system", "content": self.agent_config.system_prompt}]
        
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "I'll incorporate this context."})
        
        if self.conversation_history:
            for msg in self.conversation_history[-5:]:
                messages.append(msg)
        
        messages.append({"role": "user", "content": user_input})
        return messages
    
    def _calculate_cost(self, tokens: int) -> float:
        return (tokens / 1000) * 0.03  # $0.03 per 1K tokens (example)
    
    def process_query(self, query: str, context: Optional[str] = None, 
                     params: Optional[Dict] = None) -> Dict[str, Any]:
        # 1. Input safety
        safety = self.guardrails.filter_content(query)
        if not safety["is_safe"] and safety["severity"] > 0.7:
            return {
                "error": "Input blocked due to safety policy",
                "violations": safety["violations"],
                "severity": safety["severity"]
            }
        
        safe_query = safety["filtered_response"] or query
        exec_params = ExecutionGuardrails.validate_parameters(params or {})
        
        # 2. Prepare and execute
        messages = self._prepare_messages(safe_query, context)
        
        try:
            if self.use_foundry_project:
                response = self._call_foundry_project(messages, exec_params)
                assistant_message = response["choices"][0]["message"]["content"]
                if isinstance(assistant_message, list):
                    assistant_message = "".join(
                        part.get("text", "") for part in assistant_message if isinstance(part, dict)
                    )
                elif not isinstance(assistant_message, str):
                    assistant_message = str(assistant_message)
                usage = response.get("usage", {})
                tokens = int(usage.get("total_tokens", 0))
            else:
                response = self.client.chat.completions.create(
                    model=self.azure_config.deployment,
                    messages=messages,
                    temperature=exec_params.get("temperature", 0.2),
                    max_tokens=exec_params.get("max_tokens", 350),
                    top_p=exec_params.get("top_p", 0.95),
                    seed=self.agent_config.seed,
                )
                assistant_message = response.choices[0].message.content
                tokens = response.usage.total_tokens
            
            # 3. Output safety
            output_safety = self.guardrails.filter_content(assistant_message)
            if not output_safety["is_safe"]:
                assistant_message = output_safety["filtered_response"] or "Response filtered."
            
            # 4. Update history
            self.conversation_history.append({"role": "user", "content": safe_query})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            # 5. Track metrics
            self.total_tokens += tokens
            self.requests_count += 1
            cost = self._calculate_cost(tokens)
            self.total_cost += cost
            
            return {
                "response": assistant_message,
                "tokens_used": tokens,
                "cost_estimate": cost,
                "total_cost": self.total_cost,
                "total_requests": self.requests_count,
                "safety_ok": output_safety["is_safe"],
                "violations": output_safety["violations"],
            }
            
        except Exception as e:
            error_text = str(e).strip() or "Unknown error while calling the model endpoint."
            return {"error": error_text, "message": f"Error processing query: {error_text}"}
    
    def reset(self):
        self.conversation_history = []
    
    def get_metrics(self) -> Dict:
        return {
            "requests": self.requests_count,
            "tokens": self.total_tokens,
            "cost": self.total_cost,
            "conversation_turns": len(self.conversation_history) // 2,
        }


# ============================================================================
# STREAMLIT UI
# ============================================================================

def init_session():
    """Initialize Streamlit session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "connected" not in st.session_state:
        st.session_state.connected = False


def render_chat():
    """Render chat interface"""
    st.title("🤖 AI Ops Support Agent")
    st.caption("Microsoft Foundry • Azure OpenAI • Full Guardrails")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Azure credentials
        endpoint = st.text_input("Azure Endpoint", type="password",
                                value=os.getenv("AOAI_ENDPOINT", ""))
        api_key = st.text_input("API Key", type="password",
                               value=os.getenv("AOAI_API_KEY", ""))
        deployment = st.text_input("Deployment Name",
                                  value=os.getenv("AOAI_DEPLOYMENT", ""))
        
        # Connect button
        if st.button("🔌 Connect", type="primary"):
            config = AzureOpenAIConfig(endpoint=endpoint, api_key=api_key, deployment=deployment)
            if config.validate():
                try:
                    agent_config = AgentConfig()
                    st.session_state.agent = FoundryAgent(config, agent_config)
                    st.session_state.connected = True
                    st.success("✅ Connected to Azure!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
            else:
                st.error("Please provide all credentials")
        
        # Status
        if st.session_state.connected:
            st.success("🟢 Connected")
        else:
            st.warning("🔴 Disconnected")
        
        if st.session_state.connected:
            st.divider()
            st.header("⚡ Parameters")
            
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
            max_tokens = st.slider("Max Tokens", 50, 500, 350, 25)
            top_p = st.slider("Top P", 0.5, 1.0, 0.95, 0.05)
            
            params = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
            
            st.divider()
            st.header("🛡️ Guardrails")
            st.info("Content Safety • PII Detection • Jailbreak Protection")
            
            if st.button("🔄 Reset Chat"):
                if st.session_state.agent:
                    st.session_state.agent.reset()
                    st.session_state.messages = []
                    st.success("Chat reset!")
                    st.rerun()
            
            st.divider()
            st.header("📊 Metrics")
            if st.session_state.agent:
                metrics = st.session_state.agent.get_metrics()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Requests", metrics["requests"])
                    st.metric("Tokens", f"{metrics['tokens']:,}")
                with col2:
                    st.metric("Cost", f"${metrics['cost']:.4f}")
                    st.metric("Turns", metrics["conversation_turns"])
    
    # Main chat area
    if not st.session_state.connected:
        st.info("👆 Connect to Azure using the sidebar")
        return
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Describe your ticket or ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with agent
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response = st.session_state.agent.process_query(prompt, params=params)
            
            if "error" in response:
                st.error(f"❌ {response['message']}")
                assistant_msg = f"Error: {response['message']}"
            else:
                assistant_msg = response["response"]
                st.markdown(assistant_msg)
                
                # Show metrics
                if "tokens_used" in response:
                    st.caption(f"🔄 {response['tokens_used']} tokens · ${response['cost_estimate']:.4f}")
                
                # Show warnings
                if not response.get("safety_ok", True):
                    violations = response.get("violations", [])
                    if violations:
                        st.warning(f"⚠️ Filtered: {', '.join(violations[:3])}")
            
            st.session_state.messages.append({"role": "assistant", "content": assistant_msg})


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    # Streamlit page config
    st.set_page_config(
        page_title="AI Ops Support Agent",
        page_icon="🤖",
        layout="wide",
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stChatMessage { padding: 0.5rem; margin: 0.25rem 0; }
    .status-connected { color: #00ff00; font-weight: bold; }
    .status-disconnected { color: #ff0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    init_session()
    render_chat()


# ============================================================================
# LAUNCHER
# ============================================================================

def launch_streamlit():
    """Launch the Streamlit app"""
    port = int(os.getenv("STREAMLIT_PORT", "8502"))

    print("=" * 50)
    print("🤖 AI Ops Support Agent")
    print("=" * 50)
    print("\n📋 Before starting, set these environment variables:")
    print("   export AOAI_ENDPOINT='https://your-resource.openai.azure.com/'")
    print("   export AOAI_API_KEY='your-api-key'")
    print("   export AOAI_DEPLOYMENT='gpt-4-deployment'")
    print("\n   Or create a .env file with these values.")
    print(f"\n🚀 Starting Streamlit UI on port {port}...")
    print(f"   Open http://localhost:{port} in your browser")
    print("   Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run Streamlit
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, "-m", "streamlit", "run", script_path,
        f"--server.port={port}",
        "--server.address=0.0.0.0",
        "--server.enableCORS=true",
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    # Run the app directly by default. The previous launcher path recursively
    # invoked Streamlit on the same script, which caused nested startups and port
    # conflicts. Keep a dedicated launcher flag only for explicit use.
    if len(sys.argv) > 1 and sys.argv[1] == "--launch-streamlit":
        launch_streamlit()
    else:
        main()