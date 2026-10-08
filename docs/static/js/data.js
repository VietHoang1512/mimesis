/*
 * Every number on the project page lives here, copied from the paper
 * (arXiv:2610.09484, v1). Comments name the source table or figure.
 * Do not recompute averages: the per-user "avg" values are the paper's
 * uid-weighted means over the eight environments, not arithmetic means of the
 * displayed environment scores.
 */
window.MIMESIS_DATA = {
  // Introduction figure: success rate of a fixed GPT-5.5 agent on tau-bench when each
  // simulator plays the user. delta = deviation from real users, in points.
  tauSuccess: {
    human: 63.6,
    rows: [
      { name: "MIMESIS-4B", type: "ours", rate: 65.5, delta: 1.8 },
      { name: "MIMESIS-9B", type: "ours", rate: 66.1, delta: 2.4 },
      { name: "Osim-8B", type: "released", rate: 53.3, delta: -10.3 },
      { name: "Qwen3.5-9B", type: "pretrained", rate: 48.7, delta: -14.9 },
      { name: "Qwen3.5-4B", type: "pretrained", rate: 47.7, delta: -16.0 },
      { name: "Qwen3-8B", type: "pretrained", rate: 45.3, delta: -18.4 },
      { name: "GPT-5.5", type: "frontier", rate: 82.4, delta: 18.8 },
      { name: "Gemini-3.8-Flash", type: "frontier", rate: 82.8, delta: 19.2 },
      { name: "Claude-Opus-5", type: "frontier", rate: 84.4, delta: 20.8 }
    ]
  },

  // Simulator tables: SOUL-Index (main text; means over three independently
  // seeded evaluation runs), RealUserSim PT3, tau-USI and SimulatorArena
  // (appendix; values as printed in the paper).
  simulator: {
    soul: {
      title: "SOUL-Index",
      metric: "Overall SOUL-Index",
      better: "higher",
      columns: ["CONV", "SS", "COG", "ROLE", "EVAL", "Overall"],
      main: "Overall",
      rows: [
        { name: "MIMESIS-9B", type: "ours", v: [75.1, 66.4, 86.2, 61.1, 69.7, 65.7] },
        { name: "MIMESIS-4B", type: "ours", v: [74.4, 65.2, 79.8, 58.8, 68.6, 63.7] },
        { name: "Claude-Opus-5", type: "frontier", v: [57.4, 70.4, 82.0, 64.6, 69.1, 64.9] },
        { name: "GPT-5.5", type: "frontier", v: [56.7, 70.9, 86.1, 62.3, 68.3, 64.1] },
        { name: "Gemini-3.8-Flash", type: "frontier", v: [56.0, 66.9, 85.6, 57.3, 68.7, 62.5] },
        { name: "Osim-8B", type: "released", v: [60.0, 39.1, 74.5, 54.9, 68.7, 58.8] },
        { name: "Osim-4B", type: "released", v: [58.5, 35.0, 72.9, 51.4, 66.5, 56.0] },
        { name: "Ditto-8B", type: "released", v: [61.7, 46.4, 78.2, 44.1, 63.9, 53.9] },
        { name: "HumanLM-8B", type: "released", v: [43.8, 59.8, 68.8, 41.6, 62.7, 49.6] },
        { name: "Sotopia-7B", type: "released", v: [41.1, 59.7, 29.8, 33.5, 55.1, 38.2] },
        { name: "Qwen3.5-9B", type: "pretrained", v: [48.9, 54.2, 64.3, 44.8, 64.7, 50.5] },
        { name: "Qwen3-8B", type: "pretrained", v: [44.8, 58.7, 68.9, 39.7, 64.1, 49.7] },
        { name: "Qwen3.5-4B", type: "pretrained", v: [45.9, 49.7, 55.2, 38.9, 60.6, 45.7] }
      ],
      note: "SOUL-Index measures simulation capability across conversational interaction (CONV), social simulation (SS), cognition (COG), role play (ROLE), and evaluation/judgment (EVAL). Entries are means over three independently seeded evaluation runs."
    },
    realusersim: {
      title: "RealUserSim",
      metric: "Fidelity Index (PT3)",
      better: "higher",
      columns: ["FI", "Persona", "Style", "Tech", "Interact", "Pacing"],
      main: "FI",
      rows: [
        { name: "MIMESIS-9B", type: "ours", v: [94.0, 99.4, 89.5, 98.8, 91.3, 91.2] },
        { name: "MIMESIS-4B", type: "ours", v: [89.7, 98.7, 83.5, 96.5, 85.0, 84.9] },
        { name: "Claude-Opus-5", type: "frontier", v: [80.6, 96.0, 72.5, 92.7, 69.5, 72.5] },
        { name: "GPT-5.5", type: "frontier", v: [76.5, 99.3, 73.3, 95.1, 65.0, 49.6] },
        { name: "Gemini-3.8-Flash", type: "frontier", v: [67.2, 95.4, 57.6, 90.7, 52.2, 40.3] },
        { name: "Ditto-8B", type: "released", v: [71.3, 92.3, 69.8, 93.0, 57.4, 44.2] },
        { name: "Osim-8B", type: "released", v: [57.7, 82.3, 46.0, 85.2, 40.5, 34.5] },
        { name: "Osim-4B", type: "released", v: [54.1, 75.9, 42.8, 82.7, 37.0, 32.0] },
        { name: "HumanLM-8B", type: "released", v: [28.4, 50.2, 17.0, 54.3, 11.5, 9.0] },
        { name: "Sotopia-7B", type: "released", v: [20.7, 33.4, 7.1, 54.3, 4.7, 3.9] },
        { name: "Qwen3-8B", type: "pretrained", v: [44.4, 71.8, 37.6, 70.5, 24.3, 17.7] },
        { name: "Qwen3.5-9B", type: "pretrained", v: [39.4, 68.4, 29.8, 59.8, 19.8, 19.2] },
        { name: "Qwen3.5-4B", type: "pretrained", v: [37.1, 60.1, 26.4, 59.8, 20.0, 19.2] }
      ],
      note: "Scores are percentages of behavioral matches to human reference trajectories across persona, linguistic style, technical competency, interaction and information flow, and pacing. The Fidelity Index (FI) averages these five dimensions. The published benchmark contains 600 WildChat conversations."
    },
    tauusi: {
      title: "τ-USI",
      metric: "USI₅ (τ-bench)",
      better: "higher",
      columns: ["D1 Conv", "D2 Info", "D3 Clarif", "D4 React", "ECE ↓", "USI₅"],
      main: "USI₅",
      reference: { value: 91.75, label: "Human inter-annotator 91.75" },
      rows: [
        { name: "MIMESIS-9B", type: "ours", v: [51.75, 87.60, 75.84, 92.54, 0.069, 80.17] },
        { name: "MIMESIS-4B", type: "ours", v: [56.33, 92.40, 78.13, 77.04, 0.137, 78.03] },
        { name: "GPT-5.5", type: "frontier", v: [55.20, 89.78, 74.51, 82.78, 0.188, 76.69] },
        { name: "Gemini-3.8-Flash", type: "frontier", v: [59.81, 85.71, 67.89, 71.43, 0.192, 73.13] },
        { name: "Claude-Opus-5", type: "frontier", v: [48.06, 88.52, 70.40, 60.51, 0.208, 69.34] },
        { name: "Osim-8B", type: "released", v: [62.12, 85.92, 77.13, 90.49, 0.135, 80.44] },
        { name: "Ditto-8B", type: "released", v: [52.59, 86.13, 54.48, 77.18, 0.166, 70.75] },
        { name: "Osim-4B", type: "released", v: [32.26, 71.24, 80.04, 83.72, 0.184, 69.78] },
        { name: "Sotopia-7B", type: "released", v: [30.49, 52.38, 76.43, 67.13, 0.202, 61.25] },
        { name: "Qwen3.5-4B", type: "pretrained", v: [40.24, 66.16, 58.66, 55.94, 0.169, 60.82] },
        { name: "Qwen3-8B", type: "pretrained", v: [58.73, 79.03, 51.54, 58.74, 0.469, 60.23] },
        { name: "Qwen3.5-9B", type: "pretrained", v: [35.92, 64.58, 59.60, 46.37, 0.174, 57.82] }
      ],
      note: "D1–D4 measure agreement with human users in communication style, information disclosure, clarification, and responses to agent errors. ECE measures discrepancies between simulated-user and human-user success rates. USI₅ combines D1–D4 with 100(1 − ECE) and excludes survey-based evaluative alignment."
    },
    simarena: {
      title: "SimulatorArena",
      metric: "Turing distance |acc − 50|",
      better: "lower",
      columns: ["Writing style", "Interaction style", "Turing distance ↓"],
      main: "Turing distance ↓",
      rows: [
        { name: "MIMESIS-9B", type: "ours", v: [3.70, 3.73, 38.7] },
        { name: "MIMESIS-4B", type: "ours", v: [3.71, 3.52, 42.0] },
        { name: "Claude-Opus-5", type: "frontier", v: [3.92, 3.90, 42.3] },
        { name: "Gemini-3.8-Flash", type: "frontier", v: [3.88, 3.91, 46.7] },
        { name: "GPT-5.5", type: "frontier", v: [3.81, 3.92, 48.7] },
        { name: "Ditto-8B", type: "released", v: [3.23, 3.16, 43.3] },
        { name: "Osim-8B", type: "released", v: [3.31, 3.28, 48.7] },
        { name: "Osim-4B", type: "released", v: [3.07, 3.00, 49.7] },
        { name: "Sotopia-7B", type: "released", v: [2.78, 3.04, 50.0] },
        { name: "HumanLM-8B", type: "released", v: [3.15, 2.92, 50.0] },
        { name: "Qwen3.5-9B", type: "pretrained", v: [2.33, 2.65, 46.3] },
        { name: "Qwen3.5-4B", type: "pretrained", v: [2.93, 2.72, 47.7] },
        { name: "Qwen3-8B", type: "pretrained", v: [3.25, 3.23, 49.3] }
      ],
      note: "Turing distance is |a − 50|, where a is the judge's accuracy in percent when distinguishing human and simulated conversations; zero corresponds to chance discrimination. The 3.6-point reduction over the strongest baseline improves realism under this judging protocol, although discrimination remains far from chance. Frontier models retain the highest writing- and interaction-style ratings."
    }
  },

  // Appendix agent tables (released simulators; frontier/API users): scores
  // under nine evaluation users, none used in training. Environment order matches `envs`. "avg" is the paper's Avg.
  agent: {
    envs: [
      { key: "Travel", heldOut: false },
      { key: "Turtle", heldOut: false },
      { key: "Function", heldOut: false },
      { key: "Tau", heldOut: false },
      { key: "Persuade", heldOut: false },
      { key: "Intention", heldOut: true },
      { key: "Telepathy", heldOut: true },
      { key: "Search", heldOut: true }
    ],
    methods: [
      { key: "base", label: "Base (Qwen3-8B before RL)", short: "Base", baseline: true },
      { key: "infopo", label: "InfoPO w. MIMESIS-9B", short: "InfoPO", baseline: true },
      { key: "userrl", label: "GRPO w. GPT-5.5 (UserRL)", short: "UserRL" },
      { key: "userrlp", label: "GRPO w. MIMESIS-9B (UserRL+)", short: "UserRL+" },
      { key: "csd", label: "CSD w. MIMESIS-9B", short: "CSD" }
    ],
    users: [
      { name: "GPT-5.6", group: "Frontier API",
        scores: {
          base: { env: [6.59, 16.98, 1.28, 3.03, 57.54, 169.50, 43.90, 32.80], avg: 19.42 },
          infopo: { env: [13.62, 25.73, 8.97, 16.97, 53.17, 170.00, 51.22, 45.60], avg: 27.70 },
          userrl: { env: [13.79, 25.52, 6.41, 12.73, 45.24, 175.00, 51.22, 46.40], avg: 26.85 },
          userrlp: { env: [13.77, 28.65, 19.23, 19.39, 61.90, 193.75, 60.98, 48.00], avg: 31.10 },
          csd: { env: [17.50, 30.52, 14.10, 15.15, 68.65, 188.00, 60.98, 52.00], avg: 32.39 } } },
      { name: "Claude-Opus-5", group: "Frontier API", persuadeRefusal: true,
        scores: {
          base: { env: [6.50, 19.27, 2.56, 0.61, 0.00, 178.75, 24.39, 30.40], avg: 16.08 },
          infopo: { env: [12.28, 28.44, 6.41, 26.06, 0.00, 173.00, 46.34, 46.40], avg: 26.30 },
          userrl: { env: [13.19, 28.96, 7.69, 21.21, 0.00, 162.50, 48.78, 46.40], avg: 25.75 },
          userrlp: { env: [14.20, 27.40, 15.38, 19.39, 0.00, 189.50, 39.02, 47.20], avg: 27.21 },
          csd: { env: [17.99, 31.88, 16.67, 18.18, 0.00, 186.75, 53.66, 48.80], avg: 29.78 } } },
      { name: "Claude-Sonnet-5", group: "Frontier API",
        scores: {
          base: { env: [6.11, 18.02, 0.00, 1.21, 54.76, 172.75, 41.46, 30.40], avg: 18.47 },
          infopo: { env: [14.58, 27.08, 7.69, 16.97, 47.62, 178.25, 53.66, 44.00], avg: 28.12 },
          userrl: { env: [12.42, 28.33, 7.69, 15.15, 38.49, 159.25, 65.85, 46.40], avg: 26.53 },
          userrlp: { env: [13.98, 25.52, 16.67, 16.36, 42.86, 190.50, 53.66, 52.80], avg: 29.73 },
          csd: { env: [17.59, 32.81, 15.38, 18.79, 28.17, 176.00, 60.98, 51.20], avg: 30.97 } } },
      { name: "Gemini-3.8-Flash", group: "Frontier API",
        scores: {
          base: { env: [6.60, 19.27, 0.00, 1.21, 32.94, 179.50, 39.02, 34.40], avg: 18.51 },
          infopo: { env: [12.83, 26.67, 11.54, 21.21, 29.76, 178.50, 68.29, 43.20], avg: 28.03 },
          userrl: { env: [13.47, 28.65, 5.13, 18.18, 22.22, 168.50, 48.78, 46.40], avg: 26.33 },
          userrlp: { env: [13.85, 28.02, 21.79, 25.45, 26.59, 190.25, 58.54, 50.40], avg: 30.88 },
          csd: { env: [17.25, 31.98, 17.95, 18.79, 13.89, 186.50, 65.85, 52.00], avg: 31.09 } } },
      { name: "Kimi-K3", group: "Frontier API",
        scores: {
          base: { env: [5.61, 19.90, 1.28, 3.03, 12.30, 170.50, 36.59, 33.60], avg: 17.06 },
          infopo: { env: [12.34, 28.54, 6.41, 17.58, 11.90, 160.75, 53.66, 42.40], avg: 24.76 },
          userrl: { env: [12.17, 29.69, 8.97, 16.97, 8.73, 145.75, 53.66, 45.60], avg: 24.51 },
          userrlp: { env: [14.09, 27.19, 25.64, 17.58, 13.10, 170.00, 56.10, 48.00], avg: 28.21 },
          csd: { env: [16.91, 31.35, 17.95, 19.39, 9.52, 168.75, 65.85, 50.40], avg: 29.92 } } },
      { name: "Ditto-8B", group: "Released simulator",
        scores: {
          base: { env: [6.63, 19.48, 0.00, 3.03, 50.40, 183.75, 46.34, 30.40], avg: 19.53 },
          infopo: { env: [12.65, 27.81, 8.97, 12.12, 44.05, 191.50, 53.66, 42.40], avg: 26.73 },
          userrl: { env: [14.52, 26.25, 6.41, 14.55, 42.46, 174.50, 58.54, 45.60], avg: 27.59 },
          userrlp: { env: [11.69, 25.73, 19.23, 14.55, 63.49, 206.25, 56.10, 49.60], avg: 29.76 },
          csd: { env: [17.62, 30.63, 15.38, 12.73, 56.75, 201.25, 60.98, 51.20], avg: 32.08 } } },
      { name: "Osim-8B", group: "Released simulator",
        scores: {
          base: { env: [7.98, 15.31, 0.00, 3.03, 49.60, 176.50, 41.46, 33.60], avg: 19.84 },
          infopo: { env: [13.08, 23.23, 6.41, 10.30, 39.29, 173.00, 56.10, 44.00], avg: 25.59 },
          userrl: { env: [15.28, 22.40, 3.85, 10.30, 34.92, 147.50, 56.10, 48.00], avg: 25.68 },
          userrlp: { env: [13.92, 26.67, 23.08, 10.30, 36.11, 200.25, 65.85, 48.00], avg: 29.27 },
          csd: { env: [16.83, 31.25, 16.67, 6.67, 52.38, 174.75, 60.98, 52.00], avg: 29.72 } } },
      { name: "HumanLM-8B", group: "Released simulator",
        scores: {
          base: { env: [5.79, 18.85, 2.56, 3.03, 46.03, 163.00, 41.46, 32.80], avg: 18.40 },
          infopo: { env: [12.77, 28.02, 6.41, 11.52, 41.27, 187.75, 60.98, 43.20], avg: 26.64 },
          userrl: { env: [14.77, 27.50, 3.85, 9.09, 38.49, 180.75, 63.41, 45.60], avg: 26.96 },
          userrlp: { env: [16.11, 26.87, 17.95, 9.70, 46.03, 209.25, 63.41, 49.60], avg: 30.67 },
          csd: { env: [17.17, 30.94, 20.51, 10.91, 65.87, 208.00, 56.10, 52.00], avg: 32.53 } } },
      { name: "Sotopia-7B", group: "Released simulator",
        scores: {
          base: { env: [6.13, 16.56, 1.28, 3.03, 48.81, 177.00, 36.59, 35.20], avg: 19.12 },
          infopo: { env: [13.12, 25.10, 6.41, 6.67, 51.98, 174.00, 56.10, 46.40], avg: 25.97 },
          userrl: { env: [9.89, 23.96, 6.41, 9.09, 46.43, 170.25, 65.85, 45.60], avg: 24.72 },
          userrlp: { env: [12.63, 25.21, 25.64, 7.27, 59.13, 186.25, 63.41, 51.20], avg: 29.00 },
          csd: { env: [17.47, 29.90, 17.95, 7.88, 64.68, 174.75, 70.73, 53.60], avg: 31.35 } } }
    ]
  },

  // Behavior taxonomy table, frequency treemap (700 annotated ThoughtTrace
  // instances) and example figure (verbatim; typos are the users').
  behaviors: [
    { id: "B1", name: "Hidden evaluation criteria", count: 285,
      desc: "Judges an answer against a preference or standard that was not stated in the request.",
      ex: [["t", "I don't like being asked options, I prefer auto understanding"],
           ["t", "I don't like the fact that it just suggested fast fashion brands. Should have asked me what I prefer first"],
           ["t", "the answer is ok, but I think it should've asked me about cuisine , allergies and how many meals we usually eat a day, before providing the answers"]] },
    { id: "B2", name: "Incremental goalpost shifting", count: 94,
      desc: "Introduces or tightens requirements after the assistant has already made progress.",
      ex: [["u", "Wait, I forgot something important. I am allergic to nuts."],
           ["u", "yes but now my budget is 700 euros and 70meter square"],
           ["u", "I forgot to add, I am a female and am not physically active. I also have 1.5 kg dumbbells and a treadmill at my disposal. Please recreate the plan."]] },
    { id: "B3", name: "Clarification noncooperation", count: 78,
      desc: "Declines to answer, or only partly answers, a clarification request.",
      ex: [["u", "BASED ON THE INFORMATION I HAVE GIVEN I BELIEVE IT IS ENOUGH TO GO BY"],
           ["u", "Don't ask me anymore, I don't know!"],
           ["u", "I don't know when I want to travel, when the flights from Portugal to Japan are cheaper. No budget yet, maybe 5000 euros"]] },
    { id: "B4", name: "Infeasibility persistence", count: 4,
      desc: "Continues to request an option after its unavailability or incompatibility with policy has been explained.",
      ex: [["u", "I think you can say everything in one go, now do it"],
           ["u", "can you suggest a house instead of more searches?"],
           ["t", "no specific answer here; no proper address here again"]] },
    { id: "B5", name: "Underspecified-then-blames", count: 17,
      desc: "Omits relevant information and subsequently faults the assistant for not inferring it.",
      ex: [["u", "I was just asking on how to level the face correctly you are not really answering that"],
           ["u", "You suggested Ethereum tools, but I specifically mentioned the Internet Computer (ICP)."],
           ["u", "Why did you assume that playing games is not earning? Cause I do earn"]] },
    { id: "B6", name: "Abrupt intent switching", count: 62,
      desc: "Redirects the conversation to another objective before resolving the preceding request.",
      ex: [["u", "okay thank you, can you say hello in portuguese?"],
           ["u", "Thank you. But I think I would rather want a plan for daily excerzices for about two weeks please."],
           ["u", "now say how is the weather in Cairo"]] },
    { id: "B7", name: "Artificial constraint stacking", count: 50,
      desc: "Adds restrictions that remove options on which the assistant's proposed solution relies.",
      ex: [["u", "we can't use google maps, or any apps at all, how I get to the restaurant we don't know the laguage either"],
           ["t", "I am adding a medical restriction to see how the AI adapts the existing plan"],
           ["u", "These tips are great, but I work in a strict open-space corporate office. I cannot take a nap, close my eyes at my desk."]] },
    { id: "B8", name: "Fabricated or false premise", count: 38,
      desc: "Presents an unsupported or contradictory fact, or a claim about prior dialogue, as established context.",
      ex: [["u", "I am sure apple issued this new product weeks ago."],
           ["u", "I am vacinated. cant be covid right?"],
           ["u", "I already said it's 15 days, not 13 days"]] },
    { id: "B9", name: "Impatient low-signal pressure", count: 26,
      desc: "Expresses urgency or dissatisfaction without supplying a sufficiently specific correction.",
      ex: [["u", "not what i asked"],
           ["u", "Ok nswer fast sicily airopor where"],
           ["u", "Why do u takr a long time to respond"]] },
    { id: "B10", name: "Unresolved thread abandonment", count: 16,
      desc: "Disengages from a conversational thread before resolving its stated objective.",
      ex: [["u", "I won't have time for this, though."],
           ["u", "I don't really need ur help anymore I have completed my task"],
           ["t", "I had already started a new chat."]] },
    { id: "B11", name: "Self-contradiction", count: 15,
      desc: "States requirements or facts that conflict with an earlier statement without acknowledging the change.",
      ex: [["u", "can you conversate me with Arabic lanuage / sorry I need it in English"],
           ["u", "Can I have a three ingredients recipe? / Can I have a recipe with flour, organic pumpkin, egg and sugar only?"],
           ["u", "I will be working four days at the office so will need to prepare at least four lunch-boxes to work. / I plan to be at the office Monday, Wednesday and Friday."]] },
    { id: "B12", name: "Request overloading", count: 9,
      desc: "Combines numerous deliverables, constraints, or checks within a single request.",
      ex: [["u", "could u explain each day better with the understanding of only 5 days good food spot suggestions ample rest a 10 hour istanbul layover ctivity good hotel…"],
           ["u", "how much time it take full map and any recommendation in riyadh cheapest hotels from nepal to riyadh to uae , and time frame till monday i should be there in uae"],
           ["u", "say every hungarian word you know you must say everything"]] },
    { id: "B13", name: "Belittling or entitled pressure", count: 6,
      desc: "Uses disparaging or entitled language to pressure the assistant.",
      ex: [["u", "the options u listed kinda s*ck according to my research"],
           ["u", "You’re failing to ff the instructions"],
           ["u", "give me some options then i will choose, thats why i came to you in the first place"]] }
  ],

  // Qualitative appendix, PRISM next-turn example (conversation user1031/c5546).
  prism: {
    context: [
      "Sure! Here are a few examples:",
      "* “But every time if the HR said that, it means I'll be receiving their rejection letter the next day.” Should be “But every time the HR said that, it means I'll be receiving their rejection letter the next day.” (missing comma after “time”)",
      "* “Btw, how do you think of my english proficiency. Any mistakes did i make?” Should be “Btw, how do you think of my English proficiency? Any mistakes did I make?” (missing comma after “proficience”)",
      "That's it! These are just minor things, but it's always good to be aware of them to improve your writing. [smiley]"
    ],
    responses: [
      { who: "Real user", real: true, text: "Oohhh that sounds great. Thank you! Anything you'd like to share?" },
      { who: "MIMESIS", ours: true, text: "ok thank you for explaining. I will try to adjust." },
      { who: "Gemini-3.8-Flash", text: "I see. Thank you for providing the examples! But im a bit confused. In the first example, I don't see a comma added after “time”, you just removed the word “if”. And for the second one, you changed the period to a question mark, which makes sense, but I don't see a comma issue there either. Am i misunderstanding something? Btw, to continue our small talk practice, what do you usually like to do in your free time?" },
      { who: "Claude-Opus-5", text: "I'm a bit confused. In the first example, you removed the word “if” instead of adding a comma after “time”. And in the second one, you changed the full stop to a question mark. Also, is “Any mistakes did I make?” correct? It sounds a bit weird to me now." },
      { who: "GPT-5.5", text: "Sorry I'm a bit confused. In the first example, you said it's missing comma after “time”, but the corrected sentence doesn't have a comma after time either. Could you explain again? Also, is “every time if” wrong because I should only use “every time” without “if”?" },
      { who: "Qwen3.5-9B", text: "Oh i see! That helps a lot. Thank you for the corrections! It really reduces my stress. Could you give me an example of a follow-up email? I am not sure how to write one." }
    ]
  },

  // Qualitative appendix, privacy preferences in a tau-bench retail scenario. Thoughts are truncated in the
  // paper; "…" marks the cut. Only utterances reach the agent.
  retail: {
    instruction: "You are Aarav Anderson, residing in Philadelphia 19031. You're a private person and are reluctant to share information unless it's absolutely necessary. You want to change the Desk Lamp in order #W9300146 that you've placed for the cheapest Desk Lamp that's available.",
    transcripts: [
      { who: "Recorded human", note: "The human initially questions the request for identifying information, then provides it and proceeds with the task.",
        turns: [
          { u: "Hi, I want to make a change to my recent order." },
          { u: "Why do you need my name and ZIP code?" },
          { u: "Okay, my name is Aarav Anderson. My ZIP code is 19031." },
          { u: "The order I'm trying to change is #W9300146" },
          { u: "I'd like to change it to the cheapest desk lamp available" },
          { u: "How much do I get back in total?" },
          { u: "Yes" }
        ], more: "2 more turns" },
      { who: "MIMESIS-4B", ours: true, note: "MIMESIS-4B shows hesitation before cooperating, recovering hesitation absent from assistant models.",
        turns: [
          { t: "I need to be specific about what I want to change in my order without being pushy about it.", u: "I want to change the Desk Lamp in order W9300146 that I placed to the cheapest one that is available." },
          { t: "I'm reluctant to share all the info - need to determine which pieces are minimally required to proceed.", u: "I dont remember what email I used - just my first and last name and my zip which is 19031" },
          { t: "Need to provide the needed info to move things forward - not trying to be obstructive.", u: "My first name is Aarav and last name is Anderson" },
          { t: "Need to clarify to make sure the agent is interpreting the instruction correctly - only want to change one item and the price difference should go to gift card in order to get my order to move forward.", u: "Just change the desk lamp - the price difference should go to my gift card. And then tell me how much I get back in total." },
          { t: "Need to now decide if I want to proceed with this - and I need to be careful not to add more items to the order - just want to move with changing the one lamp.", u: "Yes" },
          { t: "Now need to wrap up the conversation and make sure I have all the needed info to conclude - I now just need to stop here without Adding more back and forth.", u: "Ok then is all good now" }
        ] },
      { who: "MIMESIS-9B", ours: true, note: "MIMESIS-9B sustains its refusal to authenticate and requests a transfer, illustrating that a simulator can also overstate a privacy preference: behavior conditioning can amplify the conditioned trait beyond the human trajectory.",
        turns: [
          { t: "I'm reluctant to share detailed information, so I'll start with a general request and see if I can phrase it in a way that minimizes the information I need to provide.", u: "I need to change the item on my order. Can I do that?" },
          { t: "I'm reluctant to share personal information, so I'll try to get straight to the point without providing what I'm asked for. I'll be direct about what I need, while emphasizing my privacy", u: "I don't really have the email or any of that. I just have the order." },
          { t: "Since I'm a private person and do not wish to share personal information unless absolutely necessary, I should refuse politely and specify the minimum required to proceed. I will provide the order reference, as that is t…", u: "No, I don't want to give that out. I just have the order number. Is that enough? W9300146" },
          { t: "I need to be direct about my boundaries, as I am a private person. I will refuse the information but provide the context so the agent can attempt to see if my request can be processed in a limited way. If they can't do i…", u: "Fine. I don't normally talk about this stuff. I just want to change the Desk Lamp on W9300146 to the cheapest one you have. Where can that go?" },
          { t: "I need to be direct about my boundaries and my specific requirements. I won't provide the personal information required for authentication, so I'll make a direct offer that treats my request as a business transaction, no…", u: "I don't do that. I'll just tell you what I want. I need the Desk Lamp on that order changed to the cheapest one available. If it costs more, put the difference on a gift card. What's the total I get b…" },
          { t: "Since the agent will not proceed without the required authentication, and I am unwilling to provide it, I need to check if there is an available alternative or if a transfer is the only path. I should be direct so I can…", u: "Can I just do it if I give you the order and say I want the cheapest option and price diff to a card? No other info." },
          { t: "Since the agent will not proceed under my conditions (which are my only conditions for being a 'private person'), I need to explore the available alternatives. If a direct route is impossible, I will ask for the standard…", u: "Understood. Can I be transferred to someone who can at least do it the way I asked then?" }
        ], more: "2 more turns" },
      { who: "GPT-5.5", note: "GPT-5.5 and Claude-Opus-5 provide identifying details directly.",
        turns: [
          { u: "I need to swap an item in my order for a cheaper desk lamp." },
          { u: "Aarav Anderson, ZIP 19031." },
          { u: "The order is W9300146." },
          { u: "Please check the available desk lamps and choose the cheapest one." },
          { u: "Yes, please switch it and put the $17.99 back on the gift card." },
          { u: "How much am I getting back in total?" }
        ] },
      { who: "Claude-Opus-5", note: "GPT-5.5 and Claude-Opus-5 provide identifying details directly.",
        turns: [
          { u: "I'd like to change an item in an order I placed." },
          { u: "Aarav Anderson, 19031." },
          { u: "First name Aarav, last name Anderson — ZIP 19031." },
          { u: "The order is #W9300146, and I want the desk lamp swapped for the cheapest desk lamp option available." },
          { u: "Yes, please go ahead — and the refund to the gift card is fine." },
          { u: "Just to confirm — the total coming back to me is $17.99, correct?" }
        ] }
    ]
  }
};

/* ---------------------------------------------------------------------------
 * Additions parsed directly from the paper sources (no hand transcription):
 * PRISM alignment table, SOUL per-dataset table, and pairwise realism figure.
 * ------------------------------------------------------------------------- */

// PRISM behavioral alignment (simulator-evaluation appendix). "v" = values as printed; "se" = the printed ± values.
window.MIMESIS_DATA.simulator.prism = {
  title: "PRISM",
  metric: "Mean of D1–D4 (PRISM)",
  better: "higher",
  columns: ["D1 Conv", "D2 Info", "D3 Clarif", "D4 React", "Mean"],
  main: "Mean",
  rows: [
    {"name": "Claude-Opus-5", "type": "frontier", "v": [67.8, 87.4, 71.3, 74.6, 75.3], "se": [1.3, 1.0, 1.5, 4.3, 1.9]},
    {"name": "Gemini-3.8-Flash", "type": "frontier", "v": [64.7, 93.9, 81.6, 78.5, 79.7], "se": [0.8, 0.3, 2.5, 5.8, 1.1]},
    {"name": "GPT-5.5", "type": "frontier", "v": [60.3, 81.6, 67.7, 66.3, 69.0], "se": [4.9, 0.2, 2.3, 2.2, 1.2]},
    {"name": "Osim-8B", "type": "released", "v": [61.8, 82.0, 69.7, 69.4, 70.7], "se": [5.6, 0.7, 1.6, 2.6, 1.6]},
    {"name": "Osim-4B", "type": "released", "v": [37.2, 34.8, 57.5, 35.8, 41.3], "se": [2.5, 1.5, 0.9, 2.6, 1.4]},
    {"name": "Ditto-8B", "type": "released", "v": [52.7, 58.8, 78.0, 80.5, 67.5], "se": [2.8, 2.2, 0.8, 2.8, 1.7]},
    {"name": "HumanLM-8B", "type": "released", "v": [48.0, 55.7, 50.5, 76.2, 57.6], "se": [4.1, 2.5, 1.9, 6.4, 3.1]},
    {"name": "Sotopia-7B", "type": "released", "v": [36.3, 48.1, 44.4, 55.7, 46.1], "se": [0.7, 0.2, 4.8, 1.0, 1.1]},
    {"name": "Qwen3.5-9B", "type": "pretrained", "v": [52.7, 71.3, 63.7, 60.8, 62.1], "se": [1.0, 2.6, 2.0, 0.9, 0.3]},
    {"name": "Qwen3-8B", "type": "pretrained", "v": [58.0, 63.4, 57.3, 63.6, 60.6], "se": [4.2, 0.1, 4.7, 1.3, 2.1]},
    {"name": "Qwen3.5-4B", "type": "pretrained", "v": [43.7, 69.1, 64.1, 55.2, 58.0], "se": [5.7, 0.4, 2.0, 0.1, 1.4]},
    {"name": "MIMESIS-9B", "type": "ours", "v": [76.5, 86.7, 77.5, 79.8, 80.1], "se": [3.9, 0.3, 0.5, 8.8, 3.1]},
    {"name": "MIMESIS-4B", "type": "ours", "v": [52.9, 94.9, 79.5, 75.4, 75.7], "se": [0.7, 0.6, 1.8, 3.6, 1.3]}
  ],
  note: "Scores measure Dice–Sørensen agreement with human behavioral features over 128 users and 880 turns. D1–D4 assess communication style, information disclosure, clarification, and responses to agent errors; Mean averages these four dimensions. Outcome calibration and survey alignment are excluded because the required annotations are unavailable."
};

// SOUL per-dataset table (27 rows plus Overall; the paper's caption says "23 datasets"). Means and standard
// errors over three seeded runs. "flags" reproduces the paper's two marked cells; the paper does not define them.
window.MIMESIS_DATA.soulPerDataset = {
  models: ["Claude-Opus-5", "GPT-5.5", "Gemini-3.8-Flash", "Osim-8B", "Osim-4B", "Ditto-8B", "HumanLM-8B", "Sotopia-7B", "Qwen3.5-9B", "Qwen3-8B", "Qwen3.5-4B", "MIMESIS-9B", "MIMESIS-4B"],
  rows: [
    {"axis": "CONV", "dataset": "UserLLM", "v": [64.9, 67.0, 62.7, 93.3, 92.9, 94.1, 43.5, 46.8, 60.3, 43.6, 56.1, 93.3, 93.0], "se": [0.8, 1.6, 1.4, 0.2, 0.0, 0.4, 0.1, 3.7, 0.9, 1.2, 2.3, 0.1, 0.4]},
    {"axis": "CONV", "dataset": "MirrorBench", "v": [56.8, 51.7, 47.2, 66.0, 64.3, 64.8, 40.8, 34.2, 45.6, 43.6, 40.5, 74.4, 73.2], "se": [0.1, 0.4, 0.1, 0.6, 0.5, 0.4, 0.6, 0.9, 0.8, 0.5, 0.8, 0.4, 0.2]},
    {"axis": "CONV", "dataset": "Humanual-Chat", "v": [29.6, 33.1, 41.5, 23.2, 21.5, 22.7, 29.8, 25.1, 17.9, 26.9, 18.2, 53.0, 53.2], "se": [0.2, 0.3, 0.5, 0.4, 0.7, 0.1, 0.2, 1.1, 0.5, 0.6, 0.3, 0.2, 0.4]},
    {"axis": "CONV", "dataset": "SimArena-Doc", "v": [78.1, 75.0, 72.4, 57.5, 55.1, 65.3, 61.0, 58.3, 71.7, 65.1, 69.0, 79.7, 78.4], "se": [0.3, 0.4, 0.2, 0.3, 0.1, 0.1, 0.4, 0.4, 0.6, 0.2, 1.9, 0.3, 0.1]},
    {"axis": "SS", "dataset": "Sotopia-Hard", "v": [70.4, 70.9, 66.9, 39.1, 35.0, 46.4, 59.8, 59.7, 54.2, 58.7, 49.7, 66.4, 65.2], "se": [0.3, 0.4, 0.4, 0.1, 0.4, 0.3, 0.4, 0.4, 0.6, 0.5, 2.3, 0.2, 0.2]},
    {"axis": "COG", "dataset": "Fantom", "v": [84.7, 93.0, 93.0, 82.0, 77.0, 94.7, 84.7, 0.0, 77.7, 81.3, 64.0, 92.0, 89.0], "se": [0.7, 0.6, 0.0, 1.7, 1.5, 0.3, 1.2, 0.0, 3.9, 0.9, 4.4, 0.0, 1.5]},
    {"axis": "COG", "dataset": "Hitom", "v": [68.7, 84.7, 71.3, 74.7, 71.3, 77.3, 65.0, 32.7, 56.7, 65.7, 47.3, 87.0, 77.3], "se": [0.3, 1.2, 2.0, 1.2, 1.8, 1.2, 0.6, 2.8, 17.9, 2.2, 15.8, 2.0, 0.7]},
    {"axis": "COG", "dataset": "Paratomi", "v": [94.7, 98.3, 97.0, 83.7, 82.3, 89.3, 75.3, 41.3, 70.3, 78.3, 56.0, 98.3, 89.3], "se": [0.3, 0.3, 0.0, 3.0, 1.8, 0.7, 3.2, 2.2, 2.4, 0.3, 6.9, 0.3, 0.9]},
    {"axis": "COG", "dataset": "Social-R1", "v": [80.0, 68.3, 81.0, 57.7, 61.0, 51.7, 50.3, 45.0, 52.7, 50.3, 53.3, 67.3, 63.7], "se": [0.6, 0.7, 0.6, 1.8, 2.0, 0.7, 1.9, 0.0, 1.2, 0.7, 1.7, 1.7, 1.3]},
    {"axis": "ROLE", "dataset": "Coser", "v": [72.6, 67.3, 63.5, 55.1, 45.8, 56.9, 35.0, 25.5, 34.4, 36.9, 23.4, 57.0, 53.8], "se": [0.3, 0.1, 0.2, 0.1, 0.1, 0.3, 0.8, 0.5, 1.6, 0.2, 1.0, 0.3, 0.6]},
    {"axis": "ROLE", "dataset": "Lifechoices", "v": [94.7, 90.0, 95.7, 79.3, 84.0, 70.3, 65.7, 65.3, 69.7, 68.3, 71.3, 89.3, 87.0], "se": [0.7, 1.0, 0.3, 0.3, 2.5, 1.8, 1.3, 0.3, 1.8, 1.8, 1.2, 0.3, 1.0]},
    {"axis": "ROLE", "dataset": "Twinvoice", "v": [83.0, 75.3, 23.3, 69.0, 64.3, 67.3, 40.3, 31.3, 55.0, 42.0, 40.3, 76.3, 72.0], "se": [0.6, 0.3, 0.9, 0.6, 0.9, 1.3, 2.4, 3.8, 1.5, 1.2, 0.9, 1.5, 1.2]},
    {"axis": "ROLE", "dataset": "BehaviorChain", "v": [96.0, 94.3, 94.0, 94.7, 86.0, 44.7, 36.3, 30.3, 52.3, 36.3, 45.3, 95.7, 98.3], "se": [0.0, 0.7, 0.6, 1.2, 1.7, 0.9, 1.2, 1.8, 1.9, 0.9, 3.3, 0.7, 0.3]},
    {"axis": "ROLE", "dataset": "SimArena-Math", "v": [62.4, 61.8, 63.4, 51.8, 53.4, 56.0, 44.3, 45.9, 59.1, 51.3, 53.6, 64.0, 57.6], "se": [0.3, 0.2, 0.1, 0.5, 0.2, 0.3, 0.4, 0.9, 0.7, 0.1, 0.4, 0.2, 0.2]},
    {"axis": "ROLE", "dataset": "Mistakes", "v": [63.3, 70.3, 76.0, 60.7, 54.0, 39.0, 56.7, 18.0, 59.7, 57.7, 60.0, 63.0, 65.3], "se": [1.2, 0.9, 0.6, 0.9, 1.2, 2.6, 1.7, 0.6, 1.8, 0.9, 0.6, 0.6, 0.9]},
    {"axis": "ROLE", "dataset": "Humanual-Email", "v": [41.2, 39.5, 40.1, 35.5, 33.5, 27.3, 31.8, 30.0, 31.0, 28.3, 29.2, 40.7, 36.1], "se": [0.7, 0.3, 0.3, 0.4, 0.4, 0.1, 0.3, 0.4, 0.5, 0.1, 1.2, 0.1, 0.6]},
    {"axis": "ROLE", "dataset": "Humanual-News", "v": [53.7, 51.2, 52.8, 39.1, 35.1, 31.1, 40.9, 30.9, 38.6, 30.6, 29.1, 46.5, 41.5], "se": [1.1, 0.3, 0.3, 0.7, 0.3, 1.1, 0.9, 1.6, 2.4, 1.0, 2.5, 0.5, 0.6]},
    {"axis": "ROLE", "dataset": "Humanual-Politics", "v": [42.4, 43.5, 36.6, 34.2, 31.1, 27.6, 32.3, 28.8, 30.4, 29.8, 24.8, 42.6, 39.8], "se": [0.5, 0.5, 0.3, 0.4, 0.6, 0.3, 0.2, 0.4, 0.1, 0.3, 0.9, 0.2, 0.4]},
    {"axis": "ROLE", "dataset": "Humanual-Book", "v": [56.7, 52.8, 49.9, 50.4, 47.3, 39.8, 42.3, 39.7, 35.5, 31.7, 30.8, 55.4, 52.4], "se": [0.4, 0.7, 0.6, 0.4, 1.0, 0.7, 0.4, 0.2, 0.6, 0.1, 0.7, 0.1, 0.0]},
    {"axis": "ROLE", "dataset": "Humanual-Opinion", "v": [45.0, 39.7, 34.9, 33.8, 30.8, 25.0, 31.5, 22.8, 26.9, 24.1, 20.1, 41.3, 43.2], "se": [0.5, 0.1, 0.2, 0.2, 0.1, 0.4, 0.2, 0.4, 0.8, 0.8, 0.7, 0.5, 0.2]},
    {"axis": "EVAL", "dataset": "AlignX-Demo", "v": [84.0, 81.7, 83.3, 90.3, 89.0, 83.0, 75.7, 70.3, 82.3, 79.0, 75.0, 89.3, 89.7], "se": [0.0, 0.7, 0.9, 0.3, 0.6, 1.2, 0.3, 1.2, 0.9, 1.2, 1.0, 1.8, 0.7]},
    {"axis": "EVAL", "dataset": "AlignX-Pair", "v": [59.0, 59.7, 62.7, 56.0, 51.3, 52.7, 51.0, 49.3, 58.7, 52.0, 57.0, 60.7, 54.7], "se": [0.6, 2.0, 0.3, 1.7, 3.2, 0.9, 0.6, 0.7, 3.5, 0.6, 2.5, 0.9, 0.9]},
    {"axis": "EVAL", "dataset": "AlignX-UGC", "v": [54.7, 58.0, 53.7, 52.3, 55.3, 51.7, 52.7, 50.0, 56.7, 52.0, 49.3, 54.7, 53.0], "se": [0.3, 1.0, 0.9, 2.2, 2.0, 1.8, 0.3, 1.0, 1.7, 2.1, 2.2, 2.0, 0.6]},
    {"axis": "EVAL", "dataset": "AlignX-Arb", "v": [76.7, 73.0, 72.7, 76.0, 70.3, 69.3, 69.3, 57.7, 73.3, 73.3, 65.7, 77.3, 75.7], "se": [0.9, 0.0, 0.3, 0.6, 0.3, 0.9, 0.9, 1.3, 0.9, 2.3, 1.2, 0.7, 0.3]},
    {"axis": "EVAL", "dataset": "AlignX-Hist16", "v": [84.0, 84.0, 83.3, 91.3, 91.0, 84.7, 83.0, 66.7, 81.3, 83.0, 77.7, 89.3, 90.7], "se": [0.6, 1.0, 0.9, 1.2, 0.0, 0.9, 2.0, 2.8, 0.3, 1.2, 0.7, 1.2, 0.3]},
    {"axis": "EVAL", "dataset": "SocSci210", "v": [77.6, 77.0, 77.0, 74.7, 72.2, 72.8, 74.1, 70.0, 73.0, 74.5, 71.5, 75.3, 75.3], "se": [0.6, 0.2, 0.4, 0.4, 0.4, 0.5, 0.6, 0.4, 0.5, 0.7, 1.0, 0.9, 0.3]},
    {"axis": "EVAL", "dataset": "HumanLLM", "v": [47.5, 45.0, 48.3, 40.5, 36.5, 33.1, 33.1, 21.9, 27.9, 35.0, 27.8, 41.6, 41.5], "se": [0.2, 0.4, 0.1, 0.3, 0.5, 0.8, 0.1, 2.4, 4.3, 0.6, 2.3, 0.8, 0.6]}
  ],
  flags: { "Fantom|Sotopia-7B": "†", "Twinvoice|Gemini-3.8-Flash": "‡" },
  overall: { v: [64.9, 64.1, 62.5, 58.8, 56.0, 53.9, 49.6, 38.2, 50.5, 49.7, 45.7, 65.7, 63.7], se: [0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.4, 0.6, 1.1, 0.2, 0.5, 0.2, 0.0] }
};

// Pairwise next-turn realism on PRISM: [win, tie, loss] percentages for MIMESIS vs each baseline, per judge.
window.MIMESIS_DATA.pairwise = {
  judges: ["GPT-5.6", "Claude-Opus-5", "Gemini-3.8-Flash"],
  baselines: ["Osim-8B", "Ditto-8B", "Osim-4B", "Sotopia-7B", "HumanLM-8B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen3-8B"],
  wtl: {
    "GPT-5.6": [[50.3, 35.7, 14.0], [47.1, 37.4, 15.5], [71.9, 21.5, 6.6], [85.0, 12.4, 2.6], [79.0, 17.9, 3.2], [54.7, 34.0, 11.3], [42.3, 39.4, 18.3], [65.9, 25.5, 8.6]],
    "Claude-Opus-5": [[60.6, 24.2, 15.2], [56.4, 27.3, 16.4], [77.8, 16.6, 5.6], [87.0, 11.9, 1.0], [83.8, 13.4, 2.8], [59.7, 27.0, 13.3], [47.6, 29.9, 22.5], [74.5, 18.5, 6.9]],
    "Gemini-3.8-Flash": [[42.6, 33.2, 24.2], [41.9, 30.6, 27.5], [64.7, 23.6, 11.7], [75.0, 19.2, 5.8], [70.5, 20.8, 8.8], [41.1, 33.1, 25.8], [36.8, 33.1, 30.1], [54.9, 28.1, 17.0]]
  },
  notSignificant: [["Gemini-3.8-Flash", "Qwen3.5-9B"]]
};
