from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

try:
    from support_assistant.graph import run_graph
    from support_assistant.models import AskRequest, AskResponse
except ImportError:  # pragma: no cover
    from graph import run_graph
    from models import AskRequest, AskResponse

app = FastAPI(
        title="Zepto Support Assistant",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
)

CHAT_PAGE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Zepto Support</title>
    <style>
        :root { --ink: #17251f; --muted: #68766f; --paper: #f5f7f2; --panel: #ffffff; --lime: #c9f23d; --line: #dfe7df; }
        * { box-sizing: border-box; }
        body { margin: 0; min-height: 100vh; color: var(--ink); background: radial-gradient(circle at 90% 0%, #e5f4bd 0, transparent 32%), var(--paper); font-family: Georgia, 'Times New Roman', serif; }
        .shell { width: min(980px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0 24px; }
        header { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 22px; }
        .eyebrow { margin: 0 0 8px; color: #617c16; font: 700 12px/1.2 Arial, sans-serif; letter-spacing: .14em; text-transform: uppercase; }
        h1 { margin: 0; font-size: clamp(36px, 7vw, 70px); line-height: .92; letter-spacing: -2px; }
        .status { display: flex; align-items: center; gap: 8px; color: var(--muted); font: 12px Arial, sans-serif; white-space: nowrap; }
        .dot { width: 9px; height: 9px; border-radius: 50%; background: #73a514; box-shadow: 0 0 0 5px #dff0b8; }
        .chat { display: flex; min-height: 570px; flex-direction: column; overflow: hidden; background: var(--panel); border: 1px solid var(--line); box-shadow: 0 22px 60px rgba(44, 64, 35, .11); }
        .messages { flex: 1; display: flex; flex-direction: column; gap: 16px; padding: 24px; overflow-y: auto; }
        .message { max-width: min(740px, 88%); padding: 16px 18px; font-size: 18px; line-height: 1.45; }
        .assistant { align-self: flex-start; background: #f0f5e8; border-left: 4px solid var(--lime); }
        .user { align-self: flex-end; color: white; background: var(--ink); }
        .meta { margin-top: 10px; color: var(--muted); font: 11px/1.4 Arial, sans-serif; }
        .composer { display: flex; gap: 10px; padding: 16px; border-top: 1px solid var(--line); background: #fbfcf9; }
        input { flex: 1; min-width: 0; padding: 15px 16px; border: 1px solid #cad7c7; border-radius: 0; color: var(--ink); background: white; font: 16px Arial, sans-serif; outline: none; }
        input:focus { border-color: #8eaf23; box-shadow: 0 0 0 3px #eaf5bd; }
        button { border: 0; padding: 0 22px; color: var(--ink); background: var(--lime); font: 700 14px Arial, sans-serif; cursor: pointer; }
        button:disabled { opacity: .55; cursor: wait; }
        .suggestions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
        .suggestion { padding: 9px 12px; border: 1px solid #cbd9b1; color: #4c6414; background: transparent; font: 12px Arial, sans-serif; cursor: pointer; }
        @media (max-width: 600px) { .shell { width: min(100% - 20px, 980px); padding-top: 20px; } header { align-items: flex-start; flex-direction: column; gap: 14px; } .chat { min-height: calc(100vh - 150px); } .messages { padding: 16px; } .message { max-width: 94%; font-size: 16px; } .composer { padding: 10px; } button { padding: 0 16px; } }
    </style>
</head>
<body>
    <main class="shell">
        <header>
            <div><p class="eyebrow">Zepto / Policy desk</p><h1>How can we help?</h1></div>
            <div class="status"><span class="dot"></span>Local grounded assistant</div>
        </header>
        <section class="chat" aria-label="Zepto support chat">
            <div class="messages" id="messages">
                <div class="message assistant">Ask about delivery, returns, refunds, membership, tracking, cancellations, gift cards, or support hours.<div class="suggestions"><button class="suggestion" type="button">What is the delivery time?</button><button class="suggestion" type="button">How long do I have to report damage?</button></div></div>
            </div>
            <form class="composer" id="composer"><input id="query" name="query" autocomplete="off" placeholder="Ask a Zepto policy question..." aria-label="Your question"><button id="send" type="submit">Send</button></form>
        </section>
    </main>
    <script>
        const messages = document.getElementById('messages');
        const input = document.getElementById('query');
        const send = document.getElementById('send');
        function addMessage(text, kind, meta) {
            const item = document.createElement('div'); item.className = `message ${kind}`; item.textContent = text;
            if (meta) { const note = document.createElement('div'); note.className = 'meta'; note.textContent = meta; item.appendChild(note); }
            messages.appendChild(item); messages.scrollTop = messages.scrollHeight;
        }
        document.querySelectorAll('.suggestion').forEach(button => button.addEventListener('click', () => { input.value = button.textContent; input.focus(); }));
        document.getElementById('composer').addEventListener('submit', async event => {
            event.preventDefault(); const query = input.value.trim(); if (!query) return;
            addMessage(query, 'user'); input.value = ''; send.disabled = true; send.textContent = '...';
            try { const response = await fetch('/ask', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query}) });
                const data = await response.json(); if (!response.ok) throw new Error(data.detail || 'Request failed');
                const sources = data.sources.length ? `Sources: ${data.sources.join(', ')}` : 'General question'; addMessage(data.answer, 'assistant', `${sources}  ·  Confidence ${(data.confidence * 100).toFixed(0)}%`);
            } catch (error) { addMessage(`Sorry, I could not answer that: ${error.message}`, 'assistant'); }
            finally { send.disabled = false; send.textContent = 'Send'; input.focus(); }
        });
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def chat_page() -> str:
        return CHAT_PAGE


@app.post("/ask", response_model=AskResponse)
def ask_support(request: AskRequest) -> AskResponse:
    return run_graph(request.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
