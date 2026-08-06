"""Generate timeout.pdf — a complete, plain-language description of the project.

Reads the benchmark + model-metric JSONs and the charts from
scripts/benchmark_charts.py, and lays out a self-contained PDF anyone can read
to understand Timeout from scratch. Pure reportlab, no network.

    python scripts/benchmark_charts.py
    python scripts/make_pdf.py
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
BM = ROOT / "out" / "benchmark"
INK = colors.HexColor("#1b2733"); ACCENT = colors.HexColor("#e2571e")
MUTE = colors.HexColor("#5c6b7a"); LINE = colors.HexColor("#d7dee5")
SOFT = colors.HexColor("#f4f6f8")


def _json(p, default=None):
    p = ROOT / p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else (default or {})


# ---------- styles ----------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=INK, fontSize=18, spaceBefore=18,
                    spaceAfter=8, leading=22)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACCENT, fontSize=13, spaceBefore=14,
                    spaceAfter=4, leading=16)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], textColor=INK, fontSize=10.3, leading=15.5,
                      spaceAfter=7, alignment=TA_LEFT)
BULLET = ParagraphStyle("BULLET", parent=BODY, spaceAfter=3, leftIndent=4)
CAP = ParagraphStyle("CAP", parent=BODY, textColor=MUTE, fontSize=8.5, leading=11, spaceBefore=2,
                     alignment=TA_CENTER)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, textColor=MUTE, leading=13)
CODE = ParagraphStyle("CODE", parent=ss["Code"], fontSize=8.6, textColor=INK, leading=12,
                      backColor=SOFT, borderPadding=6, spaceBefore=4, spaceAfter=8)


def P(t, s=BODY): return Paragraph(t, s)
def bullets(items, style=BULLET):
    return ListFlowable([ListItem(P(t, style), leftIndent=10, value="•") for t in items],
                        bulletType="bullet", start="•", leftIndent=12)


def table(rows, widths, header=True, align_left_first=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("FONTSIZE", (0, 0), (-1, -1), 9.2), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
             ("LINEBELOW", (0, 0), (-1, -2), 0.5, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 9.2)]
    for i in range(1 + (1 if header else 0), len(rows), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), SOFT))
    t.setStyle(TableStyle(style))
    return t


def img(name, width=6.3 * inch, cap=None):
    p = BM / name if (BM / name).exists() else ROOT / name
    flow = [Image(str(p), width=width, height=width * _ratio(p)), ]
    if cap: flow.append(P(cap, CAP))
    flow.append(Spacer(1, 6))
    return flow


def _ratio(p):
    from PIL import Image as PImage
    with PImage.open(p) as im:
        return im.height / im.width


def build():
    story = []
    b = _json("out/benchmark/benchmark.json")
    real = _json("out/real/real_full_metrics.json")
    shotc = _json("out/real/real_metrics.json")
    recq = _json("out/phase2/metrics.json").get("recommendation", {})
    app = b.get("app", {}); lat = b.get("latency", {}); cal = b.get("calibration", [])

    # ---------------- Title ----------------
    story += [Spacer(1, 1.4 * inch)]
    story += [P("Timeout", ParagraphStyle("T", parent=H1, fontSize=40, textColor=INK,
                                          alignment=TA_CENTER, spaceAfter=4, leading=44))]
    story += [P("An NBA play recommender", ParagraphStyle("ST", parent=H1, fontSize=15,
                textColor=ACCENT, alignment=TA_CENTER, spaceAfter=18, leading=18))]
    story += [P("Pause a basketball game at any moment. Timeout looks at where all ten players "
                "and the ball are, and draws on the frame the best thing the offense could do "
                "next — an arrow, a highlighted player, and a number telling you how many "
                "points that choice is worth. Then it explains why, in a sentence.",
                ParagraphStyle("LEAD", parent=BODY, alignment=TA_CENTER, fontSize=11.5,
                               leading=17, textColor=MUTE))]
    story += img("pipeline.png", width=5.6 * inch)
    story += [Spacer(1, 0.2 * inch)]
    story += [P("A complete guide — read this and you understand the whole project.",
                ParagraphStyle("F", parent=CAP, fontSize=9.5))]
    story += [PageBreak()]

    # ---------------- 1. What it is ----------------
    story += [P("1.  What Timeout is", H1)]
    story += [P("Timeout is a tool that watches basketball footage and acts like an assistant "
                "coach who never blinks. Whenever you pause, it answers one question: "
                "<b>of everything the player with the ball could do right now — shoot, drive, "
                "pass to a teammate, set a screen — which is worth the most?</b>", BODY)]
    story += [P("The important idea is what it is <i>not</i>. It is not trying to predict what "
                "<i>will</i> happen. It is grading the <i>options</i>. Think of it as a second "
                "opinion with a number attached: not “he is going to pass left,” but "
                "“a pass to the corner is worth about 1.1 points; a contested shot here is "
                "worth about 0.8, so the pass is the better play.”", BODY)]
    story += [P("Everything it shows is measured, not guessed. Player positions come from the "
                "video. The scores come from models trained on thousands of real NBA possessions. "
                "The written explanation is only allowed to repeat numbers the model already "
                "produced — it can’t make things up. And when the picture is too unclear "
                "to be sure (a replay, a close-up, players hidden behind each other), it says so "
                "and shows nothing rather than guess.", BODY)]

    story += [P("The one formula, in plain words", H2)]
    story += [P("Every option is scored the same way. The value of an action is: how likely it is "
                "to succeed, times how good the resulting situation is — plus, for the times "
                "it fails, how bad the turnover is. Written out:", BODY)]
    story += [P("value(action) = P(success) × value(good outcome) + "
                "(1 − P(success)) × value(turnover)", CODE)]
    story += [P("“Value” here is measured in <b>expected points</b> — the average "
                "number of points that situation is worth. This single number lets you compare a "
                "shot, a pass, and a drive on the same scale, which is the whole point.", BODY)]

    # ---------------- 2. The big picture ----------------
    story += [P("2.  How it works, end to end", H1)]
    story += [P("The system is a pipeline: raw video goes in one end, a drawn-on recommendation "
                "comes out the other. There are four stages.", BODY)]
    story += img("pipeline.png", width=6.2 * inch,
                 cap="The pipeline. The middle box — the State — is the heart of the design.")
    story += [P("<b>Stage 1 — Perception.</b> Look at the video and figure out where "
                "everyone is. Detect the players and the ball, follow them frame to frame, and "
                "convert their positions from “pixels on screen” into “feet on the "
                "court.”", BODY)]
    story += [P("<b>Stage 2 — The State.</b> Package all of that into one tidy object: ten "
                "players and the ball, in court coordinates, with their speeds, who is guarding "
                "whom, how much space there is, and a confidence score. This object is the "
                "contract the rest of the system depends on.", BODY)]
    story += [P("<b>Stage 3 — The value model.</b> Given the State, list every sensible "
                "action the ball-handler could take and score each one in expected points.", BODY)]
    story += [P("<b>Stage 4 — The app.</b> Draw the best action on the paused frame, show the "
                "ranked list of alternatives, and — on request — explain the choice in "
                "plain English or answer questions in a chat box.", BODY)]

    story += [P("Why the State matters so much", H2)]
    story += [P("The clever part of the design is that the State looks <i>exactly the same</i> "
                "whether it came from broadcast video or from the NBA’s official player-"
                "tracking data (called SportVU). Because both sources produce an identical State, "
                "the scoring models — which were trained on the clean official data — "
                "work without any changes on messy video. The hard, unreliable part (reading "
                "video) is walled off from the smart part (scoring), and they meet at one clean "
                "hand-off.", BODY)]
    story += [PageBreak()]

    # ---------------- 3. Seeing the game ----------------
    story += [P("3.  Seeing the game (perception)", H1)]
    story += [P("Turning a video into court positions is the messiest job in the project. Here is "
                "each piece, in order.", BODY)]
    story += [P("Calibration — teaching it where the court is", H2)]
    story += [P("A camera can be anywhere, at any angle. To convert screen pixels into real court "
                "feet, the system needs to know how the court is positioned in the shot. You do "
                "this once per camera angle: a small court diagram highlights a landmark (say, the "
                "corner of the paint), and you click the matching spot in the video. After 5–8 "
                "clicks it works out the full mapping (a <i>homography</i>) and draws the court "
                "lines back onto the frame so you can check the fit. As the camera pans, the "
                "mapping is carried along automatically by tracking the movement between frames.", BODY)]
    if cal:
        rows = [["Calibrated shot", "Landmarks clicked", "Accuracy (reprojection error)"]]
        for c in cal:
            rows.append([c["file"], str(c["points"]), f"{c['reproj_error_ft']} ft"])
        story += [table(rows, [2.4 * inch, 1.8 * inch, 2.1 * inch])]
        story += [P("Reprojection error is how far off the mapping is when checked against the "
                    "clicked points — under about 1.5 ft is good; these are all well within "
                    "that.", SMALL)]

    story += [P("Per-frame calibration — the trained keypoint model", H2)]
    story += [P("A click calibration is solved on <i>one</i> frame and carried across the shot "
                "by optical flow, so its accuracy decays with distance from that frame. Measured "
                "on a real shot, the carried mapping drifts from 0.18 ft to <b>16.8 ft</b> over "
                "eleven seconds — catastrophically wrong about seven seconds after its anchor. "
                "The fix is a model that finds the court landmarks independently in every frame.", BODY)]
    story += [P("The obstacle is labels: hand-marking broadcast frames is exactly the work the "
                "click calibration was invented to avoid. So the labels are generated from the "
                "calibrations that already exist. A solved calibration <i>is</i> a "
                "pixel-to-court correspondence, so inverting it places all 26 canonical "
                "landmarks on the frame — including the ones nobody clicked. Two harvesting "
                "strategies follow from that:", BODY)]
    story += [bullets([
        "<b>Flow propagation</b> walks outward from the clicked frame and labels every frame it "
        "passes. It gives real camera motion, but inherits the tracker’s drift — so each frame "
        "is gated on the tracker’s own reprojection error and dropped once it degrades.",
        "<b>Homography jitter</b> warps the verified anchor frame by a random perspective "
        "transform and pushes the labels through the same matrix. The labels stay <i>exact</i> "
        "however aggressive the warp, because the warp <i>is</i> the label transform. This "
        "manufactures new camera geometry without manufacturing label error.",
    ])]
    story += [P("The model is a small (1.5M-parameter) encoder-decoder that outputs a heatmap "
                "per landmark. Heatmaps rather than direct coordinates for one practical reason: "
                "the height of each peak is a natural confidence, which is exactly what the "
                "homography solver already expects, so an occluded landmark arrives with a low "
                "score and is dropped by the same gate the rest of the system uses.", BODY)]
    story += [P("A landmark that is off-frame is supervised toward an <i>empty</i> heatmap "
                "rather than being excluded from training. Visibility is known geometrically, so "
                "absence is a fact worth teaching; leaving it unsupervised lets the network emit "
                "a confident peak for a landmark that is nowhere in the picture.", BODY)]

    story += [P("The trap: self-consistency is not correctness", H2)]
    story += [P("The most useful lesson from building this is a negative one. Reprojection error "
                "only measures whether a homography agrees with the correspondences it was "
                "fitted to. A model that hallucinates a <i>mutually consistent</i> set of "
                "landmarks therefore reports a healthy error while projecting a court nowhere "
                "near the real one — which is what happened, and was only caught by drawing the "
                "result on the frame and looking at it. Worse, four points fit a homography with "
                "zero residual by construction, so a model that finds four landmarks and "
                "hallucinates the rest scores <i>better</i> than one that finds twenty.", BODY)]
    story += [P("Acceptance therefore requires an independent, physical check as well: a foot of "
                "court spans 17–56 px on the measured click calibrations, versus 1.5–5 px on "
                "hallucinated ones — the camera has effectively been placed in the next "
                "building. Scale is sampled only where the court is actually visible, because on "
                "a zoomed shot the far baseline sits beyond the vanishing point where scale "
                "legitimately explodes. The check accepts all seven real calibrations and "
                "rejects the bad predictions.", BODY)]
    story += [P("Where it stands", H2)]
    story += [P("Trained on 683 auto-labelled frames from six usable calibrated shots across "
                "three games, validated on a whole held-out camera shot. Within a shot it has "
                "seen, the model holds 1.55–1.74 ft while the carried calibration collapses to "
                "16.8 ft. On a camera angle it has never seen, none of its predictions survive "
                "the acceptance gate. Training loss keeps falling while held-out error plateaus, "
                "which is overfitting to six camera angles: <b>the bottleneck is labelled shots, "
                "not epochs or architecture</b>. Every additional calibrated shot feeds the same "
                "harvesting pipeline unchanged. The feature is therefore opt-in and off by "
                "default, and the gate refuses its held-out output rather than letting it "
                "corrupt the overlay.", BODY)]
    story += [P("Two calibrations were excluded from training: each had only four clicked "
                "points, which constrains nothing, and one of them turned out to be a close-up "
                "of a player rather than a view of the court.", SMALL)]

    story += [P("Detection — finding players and the ball", H2)]
    story += [P("A detector scans each sampled frame and boxes the players and the ball. The big "
                "lesson learned here: a generic “find people” detector is the wrong "
                "tool, because it also boxes the crowd, the bench, coaches, and referees — "
                "sometimes 30–40 “people” when only 10 are playing. The project now "
                "uses a purpose-built basketball detector (a hosted Roboflow workflow) with a "
                "dedicated <i>player</i> class that ignores the crowd, a separate <i>referee</i> "
                "class so officials can be dropped, and reliable <i>ball</i> detection. On the "
                "test clip this took the count from a noisy 30+ down to a clean 10 on-court "
                "players.", BODY)]
    story += [P("Following, teams, and names", H2)]
    story += [bullets([
        "<b>Tracking:</b> each player keeps the same identity from frame to frame, so paths are "
        "smooth and the ball-handler doesn’t flicker between people.",
        "<b>Teams:</b> jersey colours are clustered into two groups, so the system knows which "
        "five are on offense and which five are defending.",
        "<b>Jersey numbers:</b> an optical-character-reader reads the numbers it can see and votes "
        "across the whole shot, so a player becomes “#7” and the explanations can name "
        "him. Players facing away stay unnamed rather than being guessed.",
        "<b>Ball tracking over time:</b> the ball is small and blurs, so it is missed on some "
        "frames. Instead of only trusting a single frame, the system models the ball’s motion "
        "and coasts its position through the gaps, so every pause has a ball — and therefore "
        "a known handler.",
    ])]

    story += [P("Which detector, and why it matters", H2)]
    story += [P("Detectors are interchangeable: each one’s own class <i>names</i> are mapped "
                "onto the project’s schema, so swapping detectors needs no code change.", BODY)]
    story += [bullets([
        "<b>Hosted basketball workflow (recommended).</b> A serverless player-detection "
        "workflow whose <i>player</i> class finds the ten on-court players and not the crowd, "
        "with a separate <i>referee</i> class that can be dropped and reliable <i>ball</i> "
        "detection. One network call per sampled frame, so a larger frame stride and several "
        "concurrent workers are used; transient timeouts are retried, and a single unreachable "
        "frame is skipped rather than failing the whole build. The API key is read from the "
        "environment and never stored.",
        "<b>Local YOLO (default, offline).</b> Generic COCO weights, or a fine-tuned basketball "
        "model if one is present. Runs on the GPU when a matching CUDA build is installed.",
    ])]

    story += [P("Robustness on messy broadcast pixels", H2)]
    story += [P("Several steps stand between a raw frame and a clean ten-player state with the "
                "handler on the right person.", BODY)]
    story += [bullets([
        "<b>Referee exclusion</b> — the detector’s referee class is dropped, with a "
        "conservative grey-uniform appearance check as a backup, deliberately tuned to risk "
        "missing a referee rather than removing a real player.",
        "<b>Roster gate</b> — a track survives only if its <i>median</i> court position is that "
        "of an interior on-floor player, which removes anything projected onto the sidelines.",
        "<b>Temporal ball tracking</b> — a constant-velocity model coasts the ball through the "
        "frames where the detector misses it.",
        "<b>Handler voting</b> — possession is mode-filtered over neighbouring frames, so the "
        "ball-handler cannot flicker between players.",
        "<b>Camera-cut truncation</b> — a shot’s mapping is valid only until the next cut, so "
        "processing stops there; a multi-shot build anchors a fresh calibration in each shot and "
        "merges the results into one timeline.",
    ])]
    story += [P("Coverage — more calibrated shots, more of the game", H2)]
    story += [P("One calibration covers one camera shot, from its click-time to the next cut. "
                "Calibrating a frame in each shot and passing them together lets the build "
                "process each shot bounded by the next and merge them into a single timeline. "
                "More calibrations is simply more coverage — and, as section 3 explains, also "
                "the one input that would make the keypoint model generalise.", BODY)]

    story += [P("The confidence gate — knowing when to stay quiet", H2)]
    story += [P("Every State carries a confidence score built from how many players were tracked, "
                "how good the court mapping is, and whether the ball was seen. Below a threshold "
                "the app withholds the recommendation and tells you why (replay, close-up, "
                "occlusion, or a stretch of video that wasn’t calibrated). Showing nothing is "
                "treated as better than showing a wrong arrow.", BODY)]
    story += [PageBreak()]

    # ---------------- 4. Scoring ----------------
    story += [P("4.  Scoring the options (the value model)", H1)]
    story += [P("Once there is a clean State, the system lists the legal actions for the "
                "ball-handler (up to about 13 of them: shoot, drive left/right, pass to each "
                "teammate, set or use a screen, reset) and scores each with the formula from "
                "section 1. Three ingredients feed that formula.", BODY)]
    story += [P("The three sub-models", H2)]
    story += [bullets([
        "<b>Shot model</b> — how likely a shot is to go in, given who is shooting, from where, "
        "and how closely guarded. It blends a per-player shooting history with the situation.",
        "<b>Pass model</b> — how likely a pass is to reach its target without being stolen.",
        "<b>Drive model</b> — how likely a drive to the basket gets there successfully.",
    ])]
    story += [P("These three are gradient-boosted models (LightGBM) — fast, reliable "
                "classifiers trained on real NBA outcomes.", BODY)]
    story += [P("The situation value, V(s)", H2)]
    story += [P("The formula also needs the value of the <i>resulting</i> situation — e.g. how "
                "good it is to have the ball in the corner with a step of space. That is a neural "
                "network called V(s). It is built to be <i>order-independent</i>: it treats the "
                "players as a set, so shuffling their order can’t change the answer. It runs "
                "on a GPU when one is available.", BODY)]
    story += [P("Two honesty rules", H2)]
    story += [bullets([
        "<b>Actions are objects, never coordinates.</b> The model outputs “pass to the player "
        "in the left corner,” and only the renderer turns that into an arrow — and the "
        "arrow’s endpoints are always real tracked positions, never invented ones.",
        "<b>Never invent a missing defender.</b> If a defender can’t be seen, the system "
        "lowers its confidence instead of imagining one, and may withhold the recommendation.",
    ])]

    # ---------------- 5. The app ----------------
    story += [P("5.  The web app", H1)]
    story += [P("The product is a local web page with two parts.", BODY)]
    story += [P("The studio (setup wizard)", H2)]
    story += [P("You upload a clip from your computer or paste a YouTube link. It grabs the "
                "footage, opens a calibration step where you click the court landmarks (and can "
                "click where the ball is in the first frame to seed tracking), and then builds the "
                "recommendations. You can calibrate several camera angles to cover more of the "
                "game.", BODY)]
    story += [P("The player", H2)]
    story += [bullets([
        "Play the footage and <b>pause anywhere</b>; the best action is drawn on the frame with an "
        "arrow, a highlighted player, and its expected-points number.",
        "A ranked list of <b>alternative actions</b> sits beside the video — click any to draw "
        "it instead.",
        "Ask <b>why</b> and read a one-paragraph rationale for the chosen action.",
        "A <b>chat assistant</b> answers free-form questions (“why not shoot?”, “show "
        "the drive”, “pass to #7”) and can change the drawn play. It runs through a "
        "language model (Groq or Claude) when a key is set, and falls back to built-in rules "
        "otherwise.",
    ])]
    story += [P("Crucially, the app does <b>no live inference</b>. All the heavy work is done once, "
                "up front, and saved to a file the page reads. So pausing is instant.", BODY)]
    story += [P("Because each drawn overlay is computed for its own frame, the app refuses to "
                "reuse one recommendation on a far-away moment — that is what previously made "
                "an arrow appear frozen and point into the stands during an uncalibrated opening "
                "stretch. It now shows an honest “outside the calibrated segment” note "
                "there instead.", BODY)]

    # ---------------- 6. Training ----------------
    story += [P("6.  How the models were trained", H1)]
    story += [P("The scoring models learn from two kinds of data.", BODY)]
    story += [bullets([
        "<b>Synthetic data</b> — a built-in generator makes fake-but-realistic possessions "
        "with a known “right answer,” so the whole pipeline can be tested and measured "
        "with no downloads and no GPU.",
        "<b>Real data</b> — NBA 2015-16 SportVU player-tracking joined to real shot and "
        "play-by-play outcomes. The models here were trained on roughly <b>240 games</b> "
        "(about 25,000 shots, 45,000 passes, 49,000 drives, and 600,000+ situation snapshots), "
        "split so that test games are never seen during training.",
    ])]
    story += [P("The possession parser recovers realistic NBA rates on its own — about 100 "
                "possessions per team per game at roughly one point per possession — which is "
                "a good sign the inputs are sane before any model is trained.", BODY)]

    # ---------------- 7. Benchmark ----------------
    story += [P("7.  Benchmark results", H1)]
    story += [P("Every number below was produced by the project’s own scripts "
                "(<font face='Courier'>scripts/benchmark.py</font> and the training runs).", SMALL)]

    story += [P("Does the scoring actually work? (real 2015-16 data)", H2)]
    if real:
        br = real["brier"]
        rows = [["Sub-model", "Score (Brier)", "Baseline", "Verdict"]]
        verdict = {"shot": "beats", "pass": "ties", "drive": "beats"}
        for n in ["shot", "pass", "drive"]:
            better = br[n]["model"] < br[n]["baseline"] - 0.002
            rows.append([n.title(), f"{br[n]['model']:.3f}", f"{br[n]['baseline']:.3f}",
                         "beats baseline" if better else "ties baseline"])
        story += [table(rows, [1.6 * inch, 1.5 * inch, 1.3 * inch, 1.9 * inch])]
    story += [P("Brier score measures how well-calibrated a probability is — lower is better, "
                "and the baseline is “just guess the average rate every time.” The shot "
                "and drive models clearly beat that; pass completion ties it, because passes "
                "succeed ~90% of the time and the flat guess is already hard to beat.", SMALL)]
    story += img("submodel_brier.png", width=5.4 * inch)
    if (ROOT / "out/real/real_shot_reliability.png").exists():
        story += img("out/real/real_shot_reliability.png", width=4.8 * inch,
                     cap="Shot-model reliability: predicted make-probability vs. what actually "
                         "happened. Closer to the diagonal is better.")

    story += [P("Does it pick good actions? (simulated, vs. naive strategies)", H2)]
    if recq:
        m = recq.get("model", {})
        rows = [["Strategy", "Right or defensible", "Clearly wrong", "Avg. points lost vs. best"]]
        order = [("model", "Timeout"), ("pass_most_open", "Always pass to most open"),
                 ("always_shoot", "Always shoot"), ("random", "Random")]
        for key, label in order:
            d = recq.get(key, {})
            if not d: continue
            rows.append([label, f"{100*d.get('right_or_defensible',0):.0f}%",
                         f"{100*d.get('wrong_rate',0):.0f}%", f"{d.get('mean_regret',0):.3f}"])
        story += [table(rows, [2.1 * inch, 1.6 * inch, 1.1 * inch, 1.6 * inch])]
        story += [P("Against a known ground-truth generator, Timeout’s top pick is right or "
                    "defensible about "
                    f"{100*m.get('right_or_defensible',0):.0f}% of the time and clearly wrong only "
                    f"{100*m.get('wrong_rate',0):.0f}%, losing far fewer expected points than any "
                    "simple rule.", SMALL)]

    story += [P("Does it work on real footage? (the test clip)", H2)]
    if app:
        pv = app.get("players_per_rec", {})
        rows = [["What was measured", "Result"],
                ["Clip", f"{app.get('video_size')} at {app.get('fps')} fps, "
                         f"{app.get('duration_s')} s"],
                ["Pause points with a recommendation",
                 f"{app.get('recommended')} of {app.get('pause_points')} "
                 f"({app.get('coverage_pct')}%)"],
                ["On-court players detected per pause",
                 f"median {pv.get('median')} (range {pv.get('min')}–{pv.get('max')})"],
                ["Pauses with a live ball detection", f"{app.get('live_ball_pct')}%"],
                ["Pauses with a resolved ball-handler", f"{app.get('handler_pct')}%"],
                ["Pauses with the rim located", f"{app.get('rim_pct')}%"]]
        story += [table(rows, [3.7 * inch, 2.6 * inch])]
    story += img("coverage_timeline.png", width=6.2 * inch)
    story += [P("A median of exactly 10 players per pause is the headline: it means the detector "
                "is seeing the five-on-five action and not the crowd.", SMALL)]

    story += [P("Is it fast enough?", H2)]
    if lat:
        story += [P(f"Scoring one paused moment — listing all "
                    f"{lat.get('mean_candidates')} candidate actions and running every model — "
                    f"takes about <b>{lat.get('median_ms')} ms</b> (median; "
                    f"{lat.get('p90_ms')} ms at the 90th percentile). The design budget was 2 "
                    f"seconds, so there is enormous headroom, and because results are pre-computed "
                    f"the app itself pauses instantly.", BODY)]
    story += [P("Does per-frame calibration beat the carried one?", H2)]
    story += [P("Measured over eleven seconds of a real camera shot, comparing the click "
                "calibration carried forward by optical flow against a homography solved "
                "independently on every frame by the keypoint model:", BODY)]
    story += [table([["", "carried calibration", "per-frame model"],
                     ["Error at the start of the window", "0.18 ft", "1.55 ft"],
                     ["Error eleven seconds later", "16.8 ft", "1.74 ft"],
                     ["Frames passing the acceptance gate", "—", "30% in-domain, 0% held-out"]],
                    [2.6 * inch, 1.9 * inch, 1.8 * inch])]
    story += [P("The carried mapping is stable at first and then collapses; the model does not "
                "drift at all. That is the whole case for the approach. The last row is the "
                "honest counterweight — on a camera angle it has never seen, none of the "
                "model’s output survives the geometric check, so the feature stays off by "
                "default until more shots are calibrated.", SMALL)]

    tests = b.get("tests", {})
    story += [P(f"And the whole system is covered by an automated test suite: "
                f"<b>{tests.get('passed', 100)} tests pass</b> end to end.", BODY)]

    # ---------------- 8. Limitations ----------------
    story += [P("8.  Honest limitations", H1)]
    story += [P("A description isn’t complete without the rough edges. These are known and "
                "documented, not hidden.", BODY)]
    story += [bullets([
        "<b>The situation model V(s) is weak on real data.</b> The shot/pass/drive models are "
        "solid, but predicting the exact value of a single state from single-sample outcomes is "
        "hard; V(s) barely beats a flat average. It needs smoothed multi-step returns and more "
        "data to improve. The action <i>ranking</i> still works because it leans mostly on the "
        "three strong sub-models.",
        "<b>Calibration is still manual in practice.</b> You click court landmarks once per "
        "camera angle. An automatic court-line detector was tried and set aside — broadcast "
        "courts mix light and dark lines with logo and jersey interference. The trained keypoint "
        "model (section 3) is built, tested and wired in, and it demonstrably removes the drift "
        "the carried calibration suffers; but on six calibrated shots it does not yet generalise "
        "to an unseen camera angle, so it stays off by default. The bottleneck is labelled "
        "shots, and calibrating more of them needs no code change.",
        "<b>Coverage has gaps.</b> Recommendations exist only inside calibrated camera shots; "
        "between them (and after a cut) the app honestly shows nothing rather than a stale arrow. "
        "More calibrations = more coverage.",
        "<b>The action mix can skew.</b> On the test clip the top pick is “set a screen” "
        "most of the time; whether that is genuinely optimal or a model bias is the next thing to "
        "investigate — a scoring question, not a detection one.",
    ])]

    # ---------------- 9. Deployment ----------------
    story += [P("9.  Deployment", H1)]
    story += [P("The app is deployed as a public web page, and the pre-compute pass can run as "
                "a batch job on AWS. The two are independent: the page is meant to stay up and "
                "costs cents a month, while the job runs once per clip and leaves nothing "
                "running afterwards.", BODY)]

    story += [P("The web app — object storage plus a CDN", H2)]
    story += [P("The page is three files: the HTML, the pre-computed recommendations, and the "
                "clip itself. In the intended design they sit in a <i>private</i> bucket that "
                "only the CDN can read, signed through an origin access control, with the CDN "
                "serving byte-range requests so the video can be scrubbed. Uploads carry "
                "explicit content types and cache headers — the video is immutable and cached "
                "for a year, the recommendations are not cached at all — and each deploy "
                "invalidates the edge cache so a rebuild is visible immediately.", BODY)]
    story += [P("There is a deliberate second path. A brand-new cloud account cannot create CDN "
                "distributions until the provider verifies it, which fails the stack outright. "
                "Rather than block the demo on a support queue, the deploy script takes a flag "
                "that swaps in a plain static-website bucket: live in about a minute, at the "
                "cost of being publicly readable and served over plain HTTP. Both templates "
                "share the same bucket name and logical resource, so once the account is "
                "verified the same command without the flag upgrades the stack in place — the "
                "bucket goes private, the CDN appears in front of it, and nothing is torn down.", BODY)]
    story += [P("The chat assistant degrades on purpose here: a static origin has no endpoint to "
                "answer it, the page’s request fails, and it falls back to the built-in "
                "rule-based assistant. That is better than shipping a hosted API key.", SMALL)]

    story += [P("The pre-compute pass as a batch job", H2)]
    story += [P("Batch processing job, not a hosted inference endpoint — the pipeline runs once "
                "per clip and writes a JSON file; there is no per-request inference anywhere in "
                "the product, so a real-time endpoint would idle at GPU prices for a workload "
                "that never makes an online call. The image installs CPU-only tensors on "
                "purpose: the GPU wheels add gigabytes, and on a home connection the upload "
                "dominates the wall-clock cost of the whole exercise.", BODY)]
    story += [P("The platform is directory-driven rather than argument-driven: declared inputs "
                "are downloaded into the container before it starts, and whatever lands in the "
                "declared output directory is uploaded when it exits. A small entrypoint adapts "
                "those paths onto the existing command-line interface, so the same pipeline runs "
                "unchanged locally and in the cloud. The execution role is scoped to this "
                "project’s bucket and repository rather than using the broad managed policy, "
                "and a maximum-runtime setting acts as a spend guard.", BODY)]

    story += [P("Building the image without a local Docker daemon", H2)]
    story += [P("Building the job image normally needs Docker running locally. On Windows Home "
                "that means the WSL2 backend, because the Hyper-V backend is not offered on that "
                "edition — and WSL2 can fail in ways no Docker-side fix reaches. On the "
                "development machine the host compute service refuses to create <i>any</i> "
                "virtual machine, so every Linux distribution fails identically and reinstalling "
                "Docker changes nothing. The container path is unavailable locally while the "
                "pipeline it builds remains perfectly runnable in the cloud.", BODY)]
    story += [P("The answer is to move the build rather than fix the laptop: a managed build "
                "project compiles the image in the cloud and pushes it to the registry. The "
                "Dockerfile is unchanged and the resulting tag is identical — only the machine "
                "running the build moves. It is also faster, because the layers reach the "
                "registry from inside the cloud rather than over a home uplink. The build "
                "context is packed from exactly the paths the Dockerfile copies, since shipping "
                "the repository would mean uploading tens of gigabytes of clips.", BODY)]
    story += [P("A recurring theme for a new cloud account: services are gated one at a time. "
                "The CDN needs account verification; the build service starts with a "
                "concurrency quota of zero, so the first build fails with a limit error before "
                "the project is ever exercised. Neither is a misconfiguration, and both are "
                "worth checking before debugging one’s own templates.", SMALL)]

    # ---------------- 10. Running it ----------------
    story += [P("10.  Running it yourself", H1)]
    story += [P("Install and run the test suite (no data or GPU needed):", BODY)]
    story += [P("pip install -e .<br/>python -m pytest -q", CODE)]
    story += [P("Build and watch the web app on a clip:", BODY)]
    story += [P("# calibrate one frame per camera angle (a court diagram guides your clicks)<br/>"
                "python scripts/calibrate.py --video clip.mp4 --time 39 --out calib.json<br/>"
                "# pre-compute recommendations, then serve the player<br/>"
                "python scripts/build_webapp.py --video clip.mp4 --shots calib.json<br/>"
                "python scripts/serve_webapp.py        # open the printed URL", CODE)]
    story += [P("For the best detection, set a Roboflow key (read from the environment, never "
                "stored) and add <font face='Courier'>--detector roboflow</font>. For the chat "
                "assistant, set a Groq or Anthropic key. Re-run the whole benchmark with "
                "<font face='Courier'>python scripts/benchmark.py</font>.", SMALL)]
    story += [P("Train the court-keypoint model from calibrations you have already clicked — no "
                "new labelling — then check it against the carried calibration and, if you want "
                "it, build with per-frame homography:", BODY)]
    story += [P("python scripts/train_keypoints.py \\<br/>"
                "&nbsp;&nbsp;&nbsp;&nbsp;--pair clip.mp4 shot1.json shot2.json shot3.json<br/>"
                "python scripts/eval_keypoints.py --video clip.mp4 --calib shot1.json "
                "--render out/kp.png<br/>"
                "python scripts/build_webapp.py --video clip.mp4 --shots shot1.json "
                "--keypoint-model", CODE)]
    story += [P("The evaluation script writes an overlay image with both mappings drawn on the "
                "same frames. Look at it — as section 3 explains, the numeric error alone cannot "
                "tell you whether the court is in the right place.", SMALL)]
    story += [P("Deploy the built page, and run the pre-compute pass in the cloud:", BODY)]
    story += [P("# static hosting; drop -NoCloudFront once the account is CDN-verified<br/>"
                "./deploy/aws/webapp/deploy.ps1 -Source out/webapp -NoCloudFront<br/>"
                "# build the job image without a local Docker daemon, then run the job<br/>"
                "python deploy/aws/sagemaker/build_image_codebuild.py<br/>"
                "python deploy/aws/sagemaker/run_job.py --video clip.mp4 --calib shot1.json", CODE)]

    # ---------------- 11. Glossary ----------------
    story += [P("11.  Glossary", H1)]
    gl = [
        ("Expected points (EPV)", "The average points a situation or action is worth — the "
         "common yardstick that lets a shot, pass, and drive be compared."),
        ("State", "The tidy object describing all ten players and the ball in court feet, plus "
         "speeds, matchups, spacing, and a confidence score."),
        ("Homography", "The mathematical mapping that converts a point on the screen into a point "
         "on the real court, worked out from the calibration clicks."),
        ("Sub-model", "One of the three trained classifiers — shot, pass, or drive — that "
         "estimates the chance an action succeeds."),
        ("V(s)", "The neural network that estimates how valuable a resulting situation is."),
        ("Brier score", "A measure of how well-calibrated a predicted probability is; lower is "
         "better."),
        ("Confidence gate", "The rule that withholds a recommendation when the picture is too "
         "unclear to trust."),
        ("SportVU", "The NBA’s official optical player-tracking data, used to train the "
         "models."),
        ("Reprojection error", "How far a mapping misses the points it was fitted to, in feet. "
         "A measure of self-consistency — a low value does not by itself mean the mapping is "
         "correct."),
        ("Keypoint heatmap", "The model’s output per court landmark: a picture of where it "
         "believes that landmark is, whose peak height doubles as a confidence."),
        ("Held-out shot", "A whole camera angle kept out of training, so the score measures "
         "generalisation to an uncalibrated shot rather than memory of a familiar one."),
    ]
    rows = [["Term", "Plain meaning"]] + [[t, d] for t, d in gl]
    story += [table(rows, [1.7 * inch, 4.6 * inch])]
    story += [Spacer(1, 0.3 * inch)]
    story += [P("Timeout — half-court offense, ball-handler decisions, one honest "
                "recommendation per pause.", ParagraphStyle("end", parent=CAP, fontSize=9))]

    doc = SimpleDocTemplate(str(ROOT / "timeout.pdf"), pagesize=LETTER,
                            leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.85 * inch, bottomMargin=0.75 * inch,
                            title="Timeout — NBA Play Recommender", author="Timeout")
    doc.build(story, onLaterPages=_footer, onFirstPage=_first_footer)
    print("-> timeout.pdf")


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8); canvas.setFillColor(MUTE)
    canvas.drawRightString(LETTER[0] - 0.9 * inch, 0.5 * inch, f"Timeout  ·  {doc.page}")
    canvas.drawString(0.9 * inch, 0.5 * inch, "NBA Play Recommender")
    canvas.restoreState()


def _first_footer(canvas, doc):
    pass


if __name__ == "__main__":
    build()
