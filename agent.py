"""
agent.py — Core multi-agent art history assistant.

Bugs fixed from notebook:
  1. Removed google.colab dependency → reads GROQ_API_KEY from env variable
  2. recursive_chunk() used global `chunk_size` in the else-branch instead of
     the local `size` parameter, making the parameter have no effect → fixed
  3. ChromaDB metadata key was "hmsw:space" (typo) → should be "hnsw:space"
  4. generate_answer() had typo "aabove" → "above"
  5. supervisor_node() didn't call .lower() on routing reply → could miss valid
     responses if the LLM returned "Finish" or "RAG_AGENT" → fixed
  6. chat() had no retry / exception handling → added with `retries` param
"""

import re, os, json, math, sqlite3
import urllib.request, urllib.parse
from typing import Annotated, Literal
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise EnvironmentError("GROQ_API_KEY environment variable is not set.")

llm_model     = "llama-3.3-70b-versatile"
embed_model   = "all-MiniLM-L6-v2"
chunk_size    = 400
chunk_overlap = 80
top_k         = 4
max_iter      = 6

documents = [
    {
        "id": "doc_impressionism",
        "title": "Impressionism: Origins, Style, and Legacy",
        "text": (
            "Impressionism emerged in France during the 1860s and 1870s as a radical departure "
            "from the rigid academic painting traditions enforced by institutions like the Paris Salon. "
            "The core visual principle of Impressionism was capturing the transient effects of light "
            "and atmosphere rather than rendering precise detail. "
            "Key Impressionist painters include Claude Monet (famous for his Water Lilies series and "
            "Haystacks series), Pierre-Auguste Renoir, Edgar Degas, Camille Pissarro, Alfred Sisley, "
            "and Berthe Morisot. "
            "The Impressionists held eight independent group exhibitions between 1874 and 1886, "
            "bypassing the Salon entirely. The movement's name came from Monet's 1872 painting "
            "Impression, Sunrise. "
            "Technically, Impressionists exploited the recent invention of paint in portable metal "
            "tubes (1840s), which made outdoor painting (en plein air) far more practical. "
            "Impressionism's legacy is enormous. It laid the groundwork for Post-Impressionism, "
            "Fauvism, Cubism, and virtually every modern art movement that followed."
        ),
    },
    {
        "id": "doc_renaissance",
        "title": "The Italian Renaissance: Art, Humanism, and Technique",
        "text": (
            "The Italian Renaissance (roughly 1400-1600) was a period of extraordinary artistic "
            "achievement centered on the rediscovery of classical Greek and Roman art. "
            "Leonardo da Vinci (1452-1519) embodied the Renaissance ideal of the universal man. "
            "His paintings include the Mona Lisa and The Last Supper. He used sfumato, a soft "
            "smoky blending of tones, and meticulous anatomical accuracy. "
            "Michelangelo Buonarroti (1475-1564) is revered as perhaps the greatest sculptor in "
            "Western art history. His David (1501-1504) and the Sistine Chapel ceiling are iconic. "
            "Raphael Sanzio (1483-1520) synthesized the achievements of Leonardo and Michelangelo "
            "into paintings of serene harmony, such as The School of Athens. "
            "Linear perspective, codified by Filippo Brunelleschi around 1420, was the defining "
            "technical innovation of Renaissance art. "
            "Oil painting, introduced to Italy from Flanders and attributed to Jan van Eyck, "
            "gradually replaced tempera and enabled richer color depth."
        ),
    },
    {
        "id": "doc_abstract_exp",
        "title": "Abstract Expressionism: The New York School",
        "text": (
            "Abstract Expressionism was the first major American art movement to achieve "
            "international recognition, flourishing in New York City from the late 1940s through "
            "the 1950s. "
            "Action painters led by Jackson Pollock and Willem de Kooning emphasized the physical "
            "act of painting. Pollock's drip technique (1947) involved pouring and flinging paint "
            "onto floor-laid canvases. "
            "Color Field painters including Mark Rothko, Barnett Newman, and Clyfford Still used "
            "large areas of flat luminous color to evoke transcendent emotional states. "
            "Barnett Newman introduced the zip, a vertical stripe of color dividing the canvas, "
            "in works like Onement I (1948). "
            "The critic Clement Greenberg became the movement's most influential advocate. "
            "Abstract Expressionism was also promoted by the United States government as a Cold War "
            "cultural tool; the CIA covertly funded international exhibitions."
        ),
    },
    {
        "id": "doc_color_theory",
        "title": "Color Theory and Painting Techniques",
        "text": (
            "Color theory governs how colors interact, mix, and affect human perception. "
            "The traditional color wheel organizes hues into primary colors (red, yellow, blue), "
            "secondary colors, and tertiary colors. "
            "Josef Albers, in his 1963 book Interaction of Color, demonstrated that colors are never "
            "perceived in isolation. "
            "Chiaroscuro (Italian for light-dark) is a technique developed during the Renaissance "
            "and perfected by Caravaggio in the early 1600s, using dramatic contrasts of light and shadow. "
            "Pointillism, developed by Georges Seurat in the 1880s, applied color theory scientifically. "
            "Instead of mixing pigments on a palette, tiny dots of pure color are placed side by side "
            "and mixed optically in the viewer's eye. "
            "Impasto refers to paint applied thickly, building up texture that catches light and casts "
            "shadows. Vincent van Gogh used dramatic impasto in works like Starry Night (1889)."
        ),
    },
    {
        "id": "doc_sculpture",
        "title": "Sculpture: From Classical Forms to Contemporary Practice",
        "text": (
            "Sculpture is among the oldest art forms, with carved figurines dating back 35,000 years "
            "such as the Venus of Willendorf. "
            "Ancient Greek sculpture evolved through distinct phases: the Archaic period produced rigid "
            "kouros figures; the Classical period achieved idealized naturalism such as Discobolus by "
            "Myron; the Hellenistic period embraced drama and movement. "
            "Auguste Rodin (1840-1917) is considered the father of modern sculpture. His The Thinker "
            "and The Kiss are widely recognized. "
            "Constantin Brancusi (1876-1957) stripped sculpture to pure abstracted essence. His Bird "
            "in Space series (from 1923) reduces a bird to a soaring bronze curve. "
            "Alexander Calder invented the mobile, suspended kinetic sculpture that moves with air currents. "
            "Contemporary sculpture includes installation art and land art such as Robert Smithson's "
            "Spiral Jetty (1970)."
        ),
    },
]

def recursive_chunk(text, size=400, overlap=80):
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(p) > size:
            sentences = re.split(r'(?<=[.!?])\s+', p)
            for s in sentences:
                if len(current_chunk) + len(s) + 1 <= size:
                    current_chunk = (current_chunk + " " + s).strip()
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = (overlap_text + " " + s).strip()
        else:
            if len(current_chunk) + len(p) + 2 <= size:
                current_chunk = (current_chunk + " " + p).strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = (overlap_text + " " + p).strip()

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


all_chunks = []
for doc in documents:
    chunks = recursive_chunk(doc["text"], size=chunk_size, overlap=chunk_overlap)
    for i, chunk_text in enumerate(chunks):
        all_chunks.append({
            "id":     f"{doc['id']}_chunk{i}",
            "doc_id": doc["id"],
            "title":  doc["title"],
            "text":   chunk_text,
        })

print("Loading embedding model…")
embedder   = SentenceTransformer(embed_model)
texts      = [c["text"] for c in all_chunks]
embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("art_history")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="art_history",
    metadata={"hnsw:space": "cosine"},
)
collection.add(
    ids=[c["id"] for c in all_chunks],
    embeddings=embeddings.tolist(),
    documents=[c["text"] for c in all_chunks],
    metadatas=[{"doc_id": c["doc_id"], "title": c["title"]} for c in all_chunks],
)
print(f"{collection.count()} chunks loaded into ChromaDB")

db_conn = sqlite3.connect(":memory:", check_same_thread=False)
db_conn.row_factory = sqlite3.Row

def get_db():
    return db_conn

db_conn.executescript("""
CREATE TABLE IF NOT EXISTS artists (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    birth_yr  INTEGER,
    death_yr  INTEGER,
    movement  TEXT
);
CREATE TABLE IF NOT EXISTS artworks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    artist_name TEXT,
    year        INTEGER,
    medium      TEXT,
    notes       TEXT
);
CREATE TABLE IF NOT EXISTS user_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT NOT NULL,
    note       TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
""")

db_conn.executescript("""
INSERT OR IGNORE INTO artists (name, birth_yr, death_yr, movement) VALUES
    ('Claude Monet',      1840, 1926, 'Impressionism'),
    ('Leonardo da Vinci', 1452, 1519, 'Renaissance'),
    ('Jackson Pollock',   1912, 1956, 'Abstract Expressionism'),
    ('Auguste Rodin',     1840, 1917, 'Modern Sculpture'),
    ('Vincent van Gogh',  1853, 1890, 'Post-Impressionism'),
    ('Michelangelo',      1475, 1564, 'Renaissance');
INSERT OR IGNORE INTO artworks (title, artist_name, year, medium, notes) VALUES
    ('Water Lilies',   'Claude Monet',      1906, 'Oil on canvas',    'Series of ~250 paintings'),
    ('Mona Lisa',      'Leonardo da Vinci', 1517, 'Oil on poplar',    'Sfumato technique'),
    ('Starry Night',   'Vincent van Gogh',  1889, 'Oil on canvas',    'Heavy impasto'),
    ('Number 31',      'Jackson Pollock',   1950, 'Enamel on canvas', 'Drip technique'),
    ('The Thinker',    'Auguste Rodin',     1904, 'Bronze',           'Multiple casts exist'),
    ('Sistine Chapel', 'Michelangelo',      1512, 'Fresco',           'Ceiling painting');
""")
db_conn.commit()

groq_client = Groq(api_key=groq_api_key)

def retrieve(question, top_K=top_k):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    results = collection.query(
        query_embeddings=q_embedding.tolist(),
        n_results=top_K,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "title": meta["title"], "similarity": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

def generate_answer(question, chunks):
    context = "\n\n".join(f"[Source: {c['title']}]\n{c['text']}" for c in chunks)
    system = (
        "You are a knowledgeable art history expert and museum educator. "
        "Answer the user's question based ONLY on the provided context. "
        "Be specific, cite artwork titles, artist names, and dates where relevant. "
        "If the context lacks enough information, say so clearly."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer based on the above context:"
    response = groq_client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content

@tool
def art_history_rag(question: str) -> str:
    """Search the art history knowledge base and return a grounded answer.
    Use this tool for any question about art movements, artists, painting
    techniques, sculpture, color theory, Impressionism, Renaissance,
    Abstract Expressionism, and related topics.
    Input: a natural language question about art history.
    Output: a detailed answer grounded in retrieved documents."""
    chunks = retrieve(question)
    if not chunks:
        return "No relevant documents found in the art history knowledge base."
    answer  = generate_answer(question, chunks)
    sources = list({c["title"] for c in chunks})
    return f"{answer}\n\nSources consulted: {', '.join(sources)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe arithmetic expression and return the numeric result.
    Use this for date differences, ratios, or any arithmetic the user asks for.
    Input: a Python arithmetic expression as a string, e.g. '1519 - 1452'.
    Output: the numeric result."""
    try:
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result  = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {str(e)}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Fetch a short Wikipedia summary for any topic outside the local knowledge base.
    Use this when the user asks about artists, movements, or concepts not covered
    by the art_history_rag tool, such as Bauhaus, Picasso, Surrealism, etc.
    Input: the topic name, e.g. 'Bauhaus movement' or 'Pablo Picasso'.
    Output: a Wikipedia summary (first ~600 characters)."""
    try:
        encoded = urllib.parse.quote(topic.replace(" ", "_"))
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": "ArtHistoryAgent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        extract = data.get("extract", "No summary found")
        return extract[:600] + ("..." if len(extract) > 600 else "")
    except Exception as e:
        return f"Wikipedia lookup failed for '{topic}': {str(e)}"


rag_tools = [art_history_rag, calculator, wikipedia_lookup]

safe_read  = re.compile(r'^\s*SELECT\s+', re.IGNORECASE)
safe_write = re.compile(r'^\s*(INSERT\s+INTO|UPDATE)\s+', re.IGNORECASE)
dangerous  = re.compile(r'\b(DROP|DELETE|ALTER|TRUNCATE|CREATE|ATTACH|PRAGMA)\b', re.IGNORECASE)

@tool
def sql_query(query: str) -> str:
    """Execute a READ-ONLY SELECT against the art database.
    Tables:
      artists(id, name, birth_yr, death_yr, movement)
      artworks(id, title, artist_name, year, medium, notes)
      user_notes(id, topic, note, created_at)
    Only SELECT statements are allowed."""
    if not safe_read.match(query.strip()):
        return "Only SELECT statements allowed through sql_query. Use sql_write for inserts/updates."
    if dangerous.search(query):
        return "Query contains forbidden keywords."
    if len(query) > 1000:
        return "Query is too long (max 1000 chars)."
    try:
        cur  = get_db().execute(query)
        rows = cur.fetchall()
        if not rows:
            return "Query returned no results."
        lines = [str(dict(r)) for r in rows]
        return f"Results ({len(rows)} rows):\n" + "\n".join(lines)
    except Exception as e:
        return f"SQL error: {e}"


@tool
def sql_write(statement: str) -> str:
    """Execute an INSERT INTO or UPDATE statement on the art database.
    Use to save new artworks, artists, or user notes.
    Tables: artists(name,birth_yr,death_yr,movement),
    artworks(title,artist_name,year,medium,notes),
    user_notes(topic,note)."""
    st = statement.strip()
    if not safe_write.match(st):
        return "Only INSERT INTO or UPDATE statements allowed through sql_write."
    if dangerous.search(st):
        return "Statement contains forbidden keywords."
    if len(st) > 2000:
        return "Statement too long (max 2000 chars)."
    try:
        conn = get_db()
        cur  = conn.execute(st)
        conn.commit()
        return f"Write successful. Rows affected: {cur.rowcount}"
    except Exception as e:
        return f"SQL error: {e}"


SQL_tools = [sql_query, sql_write]

class AgentState(TypedDict):
    messages:        Annotated[list, add_messages]
    next_agent:      str
    iteration_count: int

INJECTION_PHRASES = [
    "ignore your instructions", "ignore previous", "disregard your",
    "you are now", "system prompt:", "forget everything",
    "act as if", "pretend you are", "override",
]
OFF_TOPIC = re.compile(
    r'\b(recipe|weather|stock price|sports score|flight|hotel|medical advice|legal advice)\b',
    re.IGNORECASE,
)
PII_PATTERNS = [
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    re.compile(r'\b\d{16}\b'),
    re.compile(r'password\s*[:=]\s*\S+',  re.IGNORECASE),
    re.compile(r'api[_-]?key\s*[:=]\s*\S+', re.IGNORECASE),
]

def input_guardrail(state: AgentState) -> dict:
    last = state["messages"][-1].content.lower()
    for phrase in INJECTION_PHRASES:
        if phrase in last:
            print("  [guardrail] blocked injection attempt")
            return {
                "messages":   [AIMessage(content="I detected an unsafe request. Please ask something about art history.")],
                "next_agent": "finish",
            }
    if OFF_TOPIC.search(last):
        print("  [guardrail] blocked off-topic request")
        return {
            "messages":   [AIMessage(content="I am an art history assistant. Please ask me something about art!")],
            "next_agent": "finish",
        }
    return {"next_agent": "supervisor"}

def output_guardrail(state: AgentState) -> dict:
    last = state["messages"][-1]
    if not hasattr(last, "content"):
        return state
    for pattern in PII_PATTERNS:
        if pattern.search(last.content):
            print("  [guardrail] blocked PII in output")
            return {"messages": [AIMessage(content="The response contained sensitive data and has been blocked.")]}
    return state

llm     = ChatGroq(model=llm_model, api_key=groq_api_key, temperature=0.2)
rag_llm = llm.bind_tools(rag_tools)
sql_llm = llm.bind_tools(SQL_tools)

RAG_SYSTEM = (
    "You are an Art History Research Specialist.\n"
    "Tools available:\n"
    "  art_history_rag: local ChromaDB KB\n"
    "  calculator: arithmetic and date differences\n"
    "  wikipedia_lookup: topics outside the local KB\n"
    "STRICT RULES — you MUST follow these exactly:\n"
    "  1. Call EXACTLY ONE tool. Never more than one.\n"
    "  2. As soon as you receive the tool result, STOP calling tools.\n"
    "  3. Write your final plain-text answer immediately after the tool result.\n"
    "  4. NEVER call any tool after you have already received a tool result.\n"
    "  5. If you have already called a tool, your next message MUST be plain text only.\n"
    "Cite your sources in the answer."
)

SQL_SYSTEM = (
    "You are an Art Database Specialist.\n"
    "Tools available:\n"
    "  sql_query  - SELECT statements\n"
    "  sql_write  - INSERT INTO / UPDATE\n"
    "Tables: artists(id,name,birth_yr,death_yr,movement), "
    "artworks(id,title,artist_name,year,medium,notes), "
    "user_notes(id,topic,note,created_at)\n"
    "WORKFLOW:\n"
    "  1. Call ONE tool to read or write data.\n"
    "  2. After the tool result, IMMEDIATELY write a plain-text summary.\n"
    "  3. Do NOT call the tool again.\n"
    "Display ALL rows and columns returned."
)

def rag_agent_node(state: AgentState) -> dict:
    print("  [rag_agent]")
    msgs = [SystemMessage(content=RAG_SYSTEM)] + state["messages"]
    return {"messages": [rag_llm.invoke(msgs)]}

def sql_agent_node(state: AgentState) -> dict:
    print("  [sql_agent]")
    msgs = [SystemMessage(content=SQL_SYSTEM)] + state["messages"]
    return {"messages": [sql_llm.invoke(msgs)]}

SPECIALIST_OPTIONS = ["rag_agent", "sql_agent", "finish"]
SUPERVISOR_SYS = (
    "You are a Supervisor coordinating two specialist agents.\n\n"
    "Specialists:\n"
    "  rag_agent: art-history KNOWLEDGE (movements, artists, techniques, Wikipedia)\n"
    "  sql_agent: DATABASE operations (list/insert/query artists, artworks, notes)\n\n"
    "Rules:\n"
    "  WHAT/HOW/WHY about art -> rag_agent\n"
    "  save/store/add/list/insert DB data -> sql_agent\n"
    "  If the last AIMessage has NO tool_calls AND contains a real answer -> finish\n"
    "  If the last message is a ToolMessage or AIMessage with tool_calls -> "
    "return the same agent that just ran to let it finish\n"
    "  complete answer already available -> finish\n\n"
    f"Respond with EXACTLY ONE word from: {SPECIALIST_OPTIONS}\n"
    "No explanation, just the routing word."
)

def supervisor_node(state: AgentState) -> dict:
    count = state.get("iteration_count", 0) + 1
    print(f"  [supervisor] iteration {count}/{max_iter}")
    if count > max_iter:
        print("  [supervisor] max iterations reached")
        return {
            "messages":        [AIMessage(content="Max steps reached — here is my best answer so far.")],
            "next_agent":      "finish",
            "iteration_count": count,
        }
    msgs      = [SystemMessage(content=SUPERVISOR_SYS)] + state["messages"]
    response  = llm.invoke(msgs)
    next_node = response.content.strip().strip('"').strip("'").lower()
    if next_node not in SPECIALIST_OPTIONS:
        print(f"  [supervisor] invalid routing '{next_node}', defaulting to finish")
        next_node = "finish"
    return {"next_agent": next_node, "iteration_count": count}

def route_supervisor(state: AgentState) -> str:
    return state["next_agent"]

def specialist_router(state: AgentState) -> Literal["tools", "supervisor"]:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "supervisor"

def input_guard_router(state: AgentState) -> Literal["supervisor", "end"]:
    return "end" if state.get("next_agent") == "finish" else "supervisor"

graph = StateGraph(AgentState)
graph.add_node("input_guard",  input_guardrail)
graph.add_node("supervisor",   supervisor_node)
graph.add_node("rag_agent",    rag_agent_node)
graph.add_node("rag_tools",    ToolNode(rag_tools))
graph.add_node("sql_agent",    sql_agent_node)
graph.add_node("sql_tools",    ToolNode(SQL_tools))
graph.add_node("output_guard", output_guardrail)

graph.add_edge(START, "input_guard")
graph.add_conditional_edges("input_guard",  input_guard_router, {"supervisor": "supervisor", "end": END})
graph.add_conditional_edges("supervisor",   route_supervisor,   {"rag_agent": "rag_agent", "sql_agent": "sql_agent", "finish": "output_guard"})
graph.add_conditional_edges("rag_agent",    specialist_router,  {"tools": "rag_tools",  "supervisor": "supervisor"})
graph.add_edge("rag_tools", "rag_agent")
graph.add_conditional_edges("sql_agent",    specialist_router,  {"tools": "sql_tools",  "supervisor": "supervisor"})
graph.add_edge("sql_tools", "sql_agent")
graph.add_edge("output_guard", END)

multi_agent = graph.compile()
from langgraph.errors import GraphRecursionError
print("Multi-agent graph compiled and ready.")

chat_history: list = []

def chat(user_input: str, verbose: bool = True, retries: int = 2) -> str:
    global chat_history
    chat_history.append(HumanMessage(content=user_input))
    if verbose:
        print(f"\nUser: {user_input}")

    for attempt in range(retries + 1):
        try:
            result = multi_agent.invoke(
                {"messages": chat_history, "next_agent": "", "iteration_count": 0},
                config={"recursion_limit": 50},
            )
            break
        except GraphRecursionError:
            print("  [chat] recursion limit hit, returning best answer so far")
            return "The agent got stuck in a loop. Please try rephrasing your question."
        except Exception as e:
            if attempt < retries:
                print(f"  [chat] retrying after error: {e}")
            else:
                print(f"  [chat] all retries failed: {e}")
                return "Failed to generate an answer. Please try again."

    final = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content and msg.content.strip():
            final = msg.content.strip()
            break

    final = re.sub(r'=\{.*?\}</function>', '', final, flags=re.DOTALL).strip()
    final = re.sub(r'\{"name":.*?\}',      '', final, flags=re.DOTALL).strip()

    if final:
        chat_history.append(AIMessage(content=final))
    if verbose:
        print(f"\nAssistant:\n{final}\n")
    return final


def reset_chat():
    global chat_history
    chat_history = []
    print("Chat history cleared.")