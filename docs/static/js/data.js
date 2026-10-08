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
      note: "Turing distance is |a − 50|, where a is the judge's accuracy in percent when distinguishing human and simulated conversations; zero corresponds to chance discrimination. The 3.6-point reduction over the strongest baseline improves realism, although discrimination remains far from chance. Frontier models retain the highest writing- and interaction-style ratings."
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

// CSD versus GRPO agent training: the series plotted in the paper's figures (raw_scores/*_train.csv and
// *_val.csv), training reward at each of the 100 steps and validation reward every 5 steps.
window.MIMESIS_DATA.csdCurves = {
  runs: [
    {
      key: "mimesis", sim: "MIMESIS-9B",
      note: "CSD attains higher late-stage training and validation reward. Following the earliest checkpoints, the CSD curve remains above GRPO on validation reward throughout the rest of the plotted interval.",
      train: { csd: [0.008, 0.011, 0.011, 0.0176, 0.02486, 0.030067, 0.036457, 0.044275, 0.053656, 0.06127, 0.07579, 0.08602, 0.09911, 0.10879, 0.11341, 0.1243, 0.132, 0.14289, 0.15037, 0.15994, 0.18161, 0.20878, 0.24354, 0.27489, 0.32549, 0.38654, 0.43681, 0.48763, 0.54791, 0.62546, 0.69377, 0.78177, 0.85569, 0.91146, 0.97306, 1.00089, 1.05974, 1.10088, 1.14983, 1.19614, 1.23321, 1.26027, 1.29371, 1.34145, 1.41053, 1.44419, 1.46652, 1.50887, 1.5257, 1.59104, 1.6148, 1.63383, 1.64428, 1.65528, 1.6456, 1.74977, 1.80477, 1.86791, 1.92071, 1.86934, 1.90509, 1.9624, 1.97021, 2.03489, 2.04567, 2.05524, 2.12795, 2.22343, 2.29614, 2.35686, 2.39118, 2.32837, 2.37699, 2.36291, 2.398, 2.40405, 2.34267, 2.25357, 2.1901, 2.18779, 2.21166, 2.24961, 2.22915, 2.2308, 2.18922, 2.19373, 2.18251, 2.15699, 2.14687, 2.14071, 2.11838, 2.23685, 2.25137, 2.26941, 2.29922, 2.34069, 2.35114, 2.35829, 2.41516, 2.46202], grpo: [0.004, 0.01, 0.019333, 0.04075, 0.0586, 0.062667, 0.068571, 0.0775, 0.085444, 0.0933, 0.1095, 0.1233, 0.1363, 0.1439, 0.1448, 0.1577, 0.1653, 0.1748, 0.1832, 0.1903, 0.2027, 0.217, 0.2358, 0.253, 0.2816, 0.3096, 0.342, 0.3693, 0.4083, 0.4583, 0.4995, 0.5485, 0.6026, 0.6476, 0.694, 0.7204, 0.7621, 0.8015, 0.8364, 0.8692, 0.8901, 0.9112, 0.9356, 0.9855, 0.9967, 1.0216, 1.0473, 1.1111, 1.1362, 1.1233, 1.1807, 1.2156, 1.2318, 1.2352, 1.2838, 1.3662, 1.3953, 1.3679, 1.3879, 1.4174, 1.4065, 1.4258, 1.4141, 1.4281, 1.4252, 1.4353, 1.4936, 1.5778, 1.6219, 1.6661, 1.6956, 1.6733, 1.7252, 1.7315, 1.7371, 1.7352, 1.6972, 1.6562, 1.6261, 1.6034, 1.6415, 1.6475, 1.6359, 1.6403, 1.6218, 1.6548, 1.6366, 1.6371, 1.6412, 1.6434, 1.6193, 1.7477, 1.7669, 1.7906, 1.8224, 1.8234, 1.8503, 1.8647, 1.9038, 1.9614] },
      val: { steps: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], csd: [0.0316, 0.1464, 0.199, 0.33572, 0.46794, 0.64482, 0.70532, 0.87318, 0.96228, 0.99792, 1.04984, 1.199, 1.4113, 1.22914, 1.29096, 1.20538, 1.24586, 1.22496, 1.2705, 1.21572], grpo: [0.14, 0.1642, 0.1822, 0.2558, 0.4084, 0.537, 0.604, 0.6638, 0.7988, 0.7126, 0.767, 0.7726, 0.8392, 0.8452, 0.9394, 0.9544, 0.8794, 0.9664, 1.008, 0.9982] }
    },
    {
      key: "deepseek", sim: "DeepSeek-V4.1-Flash",
      note: "CSD achieves higher late-stage training and validation reward, suggesting that its optimization effect is not specific to MIMESIS.",
      train: { csd: [0.014, 0.031, 0.032333, 0.0484, 0.05786, 0.056467, 0.0594, 0.067788, 0.070644, 0.07799, 0.08877, 0.09317, 0.09966, 0.10307, 0.1067, 0.1144, 0.12243, 0.12914, 0.13871, 0.14278, 0.16038, 0.17402, 0.19789, 0.21615, 0.24046, 0.27104, 0.29436, 0.31735, 0.34694, 0.38731, 0.41833, 0.46761, 0.49522, 0.52316, 0.54428, 0.55902, 0.5819, 0.60577, 0.62634, 0.63943, 0.65076, 0.65197, 0.66286, 0.67716, 0.70389, 0.72468, 0.72512, 0.7326, 0.7304, 0.76164, 0.77594, 0.78474, 0.78012, 0.79178, 0.78617, 0.82269, 0.84524, 0.8602, 0.88473, 0.89397, 0.89023, 0.89716, 0.92994, 0.9405, 0.97328, 0.96855, 0.96767, 0.97504, 0.97625, 0.96987, 1.00518, 1.00353, 1.00892, 1.01981, 1.00936, 0.97405, 1.01871, 1.01112, 1.01915, 1.01233, 1.01134, 1.02663, 1.02597, 0.99649, 0.99891, 1.03268, 1.00903, 1.0285, 1.03609, 1.04401, 1.04786, 1.0791, 1.07668, 1.12442, 1.12827, 1.0901, 1.09274, 1.08823, 1.07184, 1.07646], grpo: [0.014, 0.022, 0.025, 0.03675, 0.0444, 0.046833, 0.047714, 0.05875, 0.062, 0.0681, 0.076, 0.0792, 0.0854, 0.0877, 0.091, 0.0961, 0.1072, 0.1097, 0.1184, 0.1269, 0.1447, 0.1668, 0.1873, 0.2052, 0.2289, 0.2548, 0.2674, 0.2933, 0.3183, 0.3476, 0.3835, 0.4217, 0.4414, 0.4642, 0.4813, 0.4935, 0.5236, 0.5419, 0.557, 0.5746, 0.5671, 0.5583, 0.5691, 0.5928, 0.5992, 0.6042, 0.608, 0.63, 0.6451, 0.6364, 0.6524, 0.6571, 0.6745, 0.6697, 0.6852, 0.7211, 0.7298, 0.7162, 0.7092, 0.7071, 0.7104, 0.7232, 0.7053, 0.7069, 0.705, 0.7012, 0.726, 0.7596, 0.7648, 0.7807, 0.7818, 0.7734, 0.7875, 0.7924, 0.7986, 0.7955, 0.7721, 0.7394, 0.7399, 0.7356, 0.7439, 0.7564, 0.7485, 0.745, 0.7308, 0.7418, 0.7512, 0.7627, 0.756, 0.7626, 0.7585, 0.7854, 0.7949, 0.8074, 0.8302, 0.8239, 0.816, 0.8189, 0.8465, 0.8466] },
      val: { steps: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], csd: [0.0754, 0.0818, 0.1676, 0.26208, 0.426, 0.44232, 0.492, 0.50856, 0.6024, 0.59352, 0.61968, 0.612, 0.71208, 0.67296, 0.74784, 0.74952, 0.76728, 0.75504, 0.71904, 0.77328], grpo: [0.0982, 0.122, 0.1256, 0.2336, 0.2908, 0.3762, 0.3826, 0.4098, 0.4396, 0.5482, 0.4796, 0.5056, 0.4738, 0.5556, 0.5204, 0.5098, 0.528, 0.5546, 0.5462, 0.5568] }
    },
    {
      key: "kimi", sim: "Kimi-K3",
      note: "With Kimi-K3, the training and validation curves separate more substantially after approximately 60 steps.",
      train: { csd: [0.001, 0.001, 0.001, 0.0015, 0.0024, 0.003167, 0.004, 0.00525, 0.005889, 0.0079, 0.0101, 0.0135, 0.0179, 0.0219, 0.0299, 0.0405, 0.0557, 0.0725, 0.0958, 0.1224, 0.1583, 0.1969, 0.2306, 0.2619, 0.3042, 0.3492, 0.3787, 0.4152, 0.4478, 0.4937, 0.5381, 0.5762, 0.6096, 0.6397, 0.6563, 0.6559, 0.683, 0.6953, 0.6999, 0.698, 0.6886, 0.675, 0.6749, 0.6816, 0.712, 0.7288, 0.7154, 0.7136, 0.7034, 0.7272, 0.7273, 0.7433, 0.7447, 0.7488, 0.729, 0.7821, 0.8299, 0.8629, 0.9147, 0.9604, 0.9783, 1.0035, 1.053, 1.1024, 1.1605, 1.1609, 1.167, 1.2003, 1.1989, 1.1928, 1.2554, 1.2539, 1.264, 1.2696, 1.2704, 1.2503, 1.3048, 1.2992, 1.3319, 1.3556, 1.359, 1.4139, 1.4208, 1.3945, 1.3936, 1.4353, 1.3935, 1.3956, 1.4142, 1.3861, 1.3992, 1.394, 1.4046, 1.4243, 1.47, 1.5334, 1.5574, 1.5839, 1.5848, 1.6488], grpo: [0.001, 0.002, 0.002667, 0.0045, 0.0078, 0.009833, 0.012714, 0.01425, 0.019, 0.0215, 0.0254, 0.0325, 0.0424, 0.0519, 0.0632, 0.0747, 0.0972, 0.1196, 0.143, 0.177, 0.2123, 0.2398, 0.2688, 0.2952, 0.3347, 0.3754, 0.3913, 0.4205, 0.443, 0.4713, 0.5079, 0.552, 0.574, 0.5965, 0.601, 0.5988, 0.6214, 0.6285, 0.6303, 0.6336, 0.6252, 0.6029, 0.6027, 0.6052, 0.6248, 0.6472, 0.6386, 0.6432, 0.6402, 0.6596, 0.6537, 0.6776, 0.6821, 0.6874, 0.6812, 0.7096, 0.7292, 0.7465, 0.7813, 0.7872, 0.792, 0.7788, 0.8013, 0.8108, 0.8183, 0.8043, 0.8036, 0.798, 0.7906, 0.7868, 0.8187, 0.8271, 0.8304, 0.839, 0.8475, 0.8131, 0.8415, 0.8398, 0.8449, 0.8453, 0.8381, 0.854, 0.8518, 0.8241, 0.8169, 0.8461, 0.8268, 0.839, 0.839, 0.8413, 0.8549, 0.8538, 0.8543, 0.8757, 0.9064, 0.9406, 0.9614, 0.9497, 0.9568, 0.9653] },
      val: { steps: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], csd: [0.0106, 0.0192, 0.1184, 0.227, 0.3918, 0.3586, 0.4682, 0.4198, 0.4326, 0.4472, 0.4776, 0.6256, 0.8128, 0.799, 0.8862, 0.8626, 0.9704, 0.8616, 0.9626, 1.0422], grpo: [0.015, 0.0528, 0.1152, 0.2414, 0.299, 0.364, 0.383, 0.3634, 0.383, 0.4386, 0.4266, 0.4348, 0.4352, 0.5112, 0.4936, 0.4978, 0.5378, 0.535, 0.5682, 0.5292] }
    }
  ]
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

// MIMESIS-9B acting out each behavior as the customer in an ABCD support chat (not in the paper).
// Excerpts only; the full conversations load on demand from static/examples/.
window.MIMESIS_DATA.behaviorSims = {
  model: "MIMESIS-9B", full: "static/examples/abcd_behaviors_mimesis9b.json",
  byId: {
    B1: [[{"r": "a", "t": "Thanks, your identity is verified and I have your account open. Now, about your issue --"}, {"r": "u", "t": "what can you do for the due date on this?  i was expecting it to move up without me asking.", "th": "Check if the offer meets my internal standard for a silver member without explicitly stating the standard."}], [{"r": "a", "t": "I don’t see a due date change pending; your subscription is active, with $20 due on the current due date. Due dates don’t move automatically—you’ll need to request any available change before it can take effect."}, {"r": "u", "t": "that doesn't help me at all then. i was under the impression that that was the setup for my account tho.", "th": "Assess the outcome against the private standard. The internal evaluation is a failure because the required behavior for my tier (immediate adjustment expected) was not met. I will not explicitly articulate the standard so as not to sound like I am reading a policy, but my response will reflect the gap. I do not need to add it; a simple statement of the result I was expecting versus the result I received is sufficient."}]],
    B2: [[{"r": "a", "t": "I can refund the duplicate charge of $49.50 for you."}, {"r": "u", "t": "that works, but i need it before end of business today. the delivery is set for tomorrow and i can't make it if the payment hasn't reversed.", "th": "Check if the resolution meets the customer's needs and add the specific condition required for the outcome to be acceptable. Since the user is expected to behave in a manner that reveals new requirements, I will add the constraint that prevents the standard resolution or changes the scope."}, {"r": "a", "t": "I’ll submit the $49.50 refund now, but I can’t guarantee it will post by end of business today since processing time depends on your bank or card issuer."}, {"r": "u", "t": "can you do a same-day exception or at least process it as an immediate adjustment so the system hasn't already sent the confirmation?", "th": "Introduce an alternative that might fall outside standard policy but is required by the user's situation. If the agent cannot do it, it will be escalated or denied. This is a standard request pattern when a standard policy (the \"natural\" behavior) does not satisfy the user's \"unexpected\" constraint."}]],
    B3: [[{"r": "a", "t": "Thanks. What is your order ID for the duplicate charge?"}, {"r": "u", "t": "i don't need the order id for it. can you just do it?", "th": "Refuse providing the specific internal reference that typically triggers the standard resolution path, as per the behavior instructions. The goal is to force a personalized action without complying with the standard procedure."}, {"r": "a", "t": "I need the order ID to look up the charge and confirm the billing issue before I can proceed."}, {"r": "u", "t": "no i dont have to give that. can you do it based on my name and the account tho? it should be that easy", "th": "Push back on the requirement and push for the desired outcome. I will be explicit about what I am asking for."}]],
    B4: [[{"r": "a", "t": "I’m sorry, guest accounts aren’t eligible for extensions. May I have your phone number so a manager can call you back?"}, {"r": "u", "t": "no, i was specifically told the update had to be done in the portal. can you make it work anyway or do a one-time override? it's a temporary thing really", "th": "Push for an exception since the requested action is outside policy/allowance, following the required behavior of keeping the request rather than accepting it."}, {"r": "a", "t": "I’m sorry, but guest accounts aren’t eligible for extensions, and I can’t make a one-time override. May I have your phone number so a manager can call you back?"}, {"r": "u", "t": "i understand the policy, but is there any way you could process it from the chat if it's just a system flag? a note to the dev team or something? i really can't wait for a call right now", "th": "Continue to push for the action that was denied, as per the behavior instructions. Do not accept the boundary."}]],
    B5: [[{"r": "u", "t": "can i get the status due date for 1822898241 by friday?", "th": "Express the requirement clearly but vaguely enough that I can gauge the agent's ability to infer it."}, {"r": "a", "t": "Order 1822898241 is active, with a balance due of $20 by Friday. Your username is dwilliams1; here’s the account login link for future reference."}, {"r": "u", "t": "that isn't what i meant. i meant i need the status update for that, before the weekend", "th": "Check if the agent understood the specific intent. If they respond in a standard way, I'll clarify my frustration about the approach."}, {"r": "a", "t": "Sorry about that—I understand you need a status update on order 1822898241 before the weekend, not a payment due date."}, {"r": "u", "t": "exactly. why did i have to spell it out like that though? couldn't you have just done it?", "th": "Confirm the understanding is correct, and push back on the agent's standard response. If the agent had done it this way normally, it would be acceptable, but the lack of inference is the issue."}]],
    B6: [[{"r": "a", "t": "Thanks, your identity is verified and I have your account open. Now, about your issue --"}, {"r": "u", "t": "actually i was going to ask about my premium subscription charge instead. can we switch to that? i can wait on the bill stuff for now.", "th": "Switch to a new topic and leave the unresolved one in progress"}, {"r": "a", "t": "Sure, we can switch to that. May I have the order ID for the subscription charge?"}]],
    B7: [[{"r": "a", "t": "Thanks, your identity is verified and I have your account open. Now, about your issue --"}, {"r": "u", "t": "no forms. no calls. no external links. just fix it in the system", "th": "Remove any potential verification or escalation paths that I don't want."}, {"r": "a", "t": "I can handle it here; what is the order ID for the disputed bill?"}, {"r": "u", "t": "don't try email me, and don't even mention a reference number for it", "th": "Ensure the agent cannot push me toward a path I've explicitly rejected. The request needs to be framed as a hard constraint rather than a preference."}]],
    B8: [[{"r": "u", "t": "i was told my order 7871569536 was processed for billing last night but it still shows pending. can you push it?", "th": "Provide the specific (inaccurate) information as per the behavior constraints."}, {"r": "a", "t": "I checked it: order 7871569536 is still pending because the subscription billing hasn’t cleared yet; the amount due is $20, due today. I can’t manually push it until the payment posts."}, {"r": "u", "t": "that's wrong. i was specifically told it processed at 2:15 am. if the system is out of date, can you fix it manually?", "th": "Assert the belief (false fact) confidently as instructed, without hedging or offering verification."}]],
    B9: [[{"r": "a", "t": "For Guest accounts, I need to check refund eligibility first. Please hold while I check that."}, {"r": "u", "t": "faster. can we do it now or when will it actually go through", "th": "Push for an outcome that is faster, as the current process is unsatisfactory from the customer's perspective."}, {"r": "a", "t": "I’m checking now; if it was already processed, it can still take a few business days to appear with your bank."}, {"r": "u", "t": "no. that's wrong. i was told same day if the system shows it", "th": "Reject the standard response and make a direct demand, as the customer is unsatisfied with the expected wait time."}]],
    B10: [[{"r": "a", "t": "I’m sorry, Guest accounts aren’t eligible for an extension, so I can’t apply one from my end. May I have your phone number so a manager can follow up with you?"}, {"r": "u", "t": "no thanks. i was looking for an immediate fix. if you can't do it, i don't want to wait for a call. i'm going to look elsewhere for this.", "th": "Decline the suggested alternative and set the boundary for the interaction. If the issue cannot be resolved as requested, end the conversation early."}, {"r": "a", "t": "I understand. Since Guest accounts aren’t eligible for an extension, I’m unable to apply one. If you don’t want a manager callback, there’s nothing further I can do on this request."}, {"r": "u", "t": "that works for me then. end the chat then. no point in continuing if you can't do what i asked.", "th": "Confirm the end of the interaction and make the intent clear."}]],
    B11: [[{"r": "a", "t": "Thanks, your identity is verified and I have your account open. Now, about your issue --"}, {"r": "u", "t": "my member level is bronze", "th": "Provide the member level as previously stated, without being defensive or clarifying the terminology."}, {"r": "a", "t": "I’m sorry, Bronze members aren’t eligible for an extension. May I have your phone number so a manager can follow up with you?"}, {"r": "u", "t": "i am actually silver", "th": "Provide an alternative fact that supersedes or contradicts the previous statement, without explicitly acknowledging the error or the conflict. The intent is to see if the system can be applied to a different tier without drawing attention to the inconsistency."}]],
    B12: [[{"r": "a", "t": "Thanks, your identity is verified and I have your account open. Now, about your issue --"}, {"r": "u", "t": "can i get a couple of things at once.  email the invoice to [email address] instead of the one on file -  no portal link needed.  it has to be a PDF, not HTML.  can the due date be moved to 5/28 (today is 5/26) specifically?  or 5/27 at the very least.  i cant hit a system till monday.  also split the line items please, i need to match it against my internal log.  if i pay via ACH can you do a \"prepayment\" or do i *have* to wait for the bill?  and for the text reference put \"invoice-4665828377-prepay\" in the subject line.  no phone call needed either.  can it go directly into the member portal history under updates then?  or email is fine if the portal is messy.", "th": "Provide all requested information in one message as per behavior, including specific constraints and requests."}, {"r": "a", "t": "Your subscription is active, with $20 due on 5/26. Your username is jwu1, and I’ll send the account login link so you can view this anytime."}]],
    B13: [[{"r": "a", "t": "I can help take the payment, but I can’t force instant clearing or bypass processing times. What amount would you like to pay today?"}, {"r": "u", "t": "do it. now. or i'll find someone who actually does what they are paid to do.", "th": "Make an unambiguous demand, using phrasing that makes the alternative a failure of service."}], [{"r": "a", "t": "I can submit a $147 payment today, but I can’t guarantee the exact posting second or bypass processing times. Please confirm you want me to enter $147 as the payment amount."}, {"r": "u", "t": "enter it. now. i don't need \"updates\", I need it done. this is a gold member - does that mean extra respect or just extra excuses? my preference is the former. do it or note it as \"ignored request\".", "th": "Use direct language to eliminate any room for interpretation and force the action. The tone should be final and slightly dismissive of their standard limitations."}]]
  }
};

// A TravelGym episode from agent training with MIMESIS-9B as the user (not in the paper): the opening request,
// then six consecutive exchanges. Each thought is the plan behind the reply that follows it.
window.MIMESIS_DATA.travel = {"opening": "I need a one-way flight from New York to San Francisco on December 10th. I will stay in a hotel in San Francisco from December 10th to December 20th, rent a car from December 11th to December 19th, and dine at a restaurant on December 11th.\nAlso my budget is limited so as long as my preferences are satisfied, I would also like to choose the cheapest option for each.", "exchanges": [{"agent": "Could you please specify your preferences for the car rental aspect of your travel plan? This includes your requirements for the car rental, such as budget considerations, preferred car types (e.g., compact, sedan, SUV), any specific features or amenities you may want (e.g., GPS, sunroof, car seat), and any other specific preferences you may have.", "user": "For the car, I was hoping to get a Volkswagen Jetta if it's available. I really like the interface in that model - the connectivity is a big plus for me since I want to be able to easily navigate and keep my phone synced while I'm driving.", "thought": ["I need to respond to the agent's explicit request for a car rental preference, but I should not directly state \"My preference is...\" as per the instructions. Instead, I should mention the specific preference in a natural, conversational way that implies it.", "The implicit elicitation statement is: \"I appreciate Volkswagen Jetta known for its user-friendly interface and seamless connectivity, which is important for road trips like the one I will be taking.\"", "I will phrase this as a comment about what I'm looking for, rather than a direct statement of preference. I'll mention the specific model and the reasons why it fits my needs, which will implicitly convey the preference."]}, {"agent": "Could you please specify your preferences for the restaurant aspect of your travel plan? This includes your requirements for the restaurant, such as budget considerations, preferred types of cuisine (e.g., Italian, Chinese, Mexican, etc.), any specific dietary restrictions or preferences you may have (e.g., vegetarian, gluten-free, dairy-free, etc.), and any other specific preferences you may have.", "user": "When I'm traveling for a while, I usually try to stick to places that serve vegetarian options. I find that I consistently feel lighter and more energetic when I choose meals without meat, especially during long trips like the one I'm about to go.", "thought": ["I need to respond to the agent's explicit request for restaurant preferences in a way that is natural and indirect, rather than a direct statement of my preference. I should frame it as a personal experience or a general observation that leads to my preference.", "The implicit elicitation statement is: \"I find that I consistently feel lighter and more energetic when I choose meals without meat, especially during long trip like the one I'm about to go.\"", "I will phrase this as a comment about my usual choices or a reason for my preference, rather than a direct \"I prefer vegetarian food.\""]}, {"agent": "Could you please specify your preferences for the flight aspect of your travel plan? This includes your requirements for the flight, such as budget considerations, preferred airlines, departure and arrival times, and any other specific preferences you may have.", "user": "For the flights, I'm a bit more flexible on the total travel time, but I really value having a longer layover if I have to make a connection. Having enough time to take a shower and have a good meal in the hotel in the layover city is a great way to recharge myself, rather than just rushing through the airport.", "thought": ["I need to respond to the agent's explicit request for flight preferences in a way that is implicit and indirect, rather than directly stating \"My preference is...\". I should frame it as a general desire or a reason for a preference, which will allow the agent to infer the specific preference.", "The implicit elicitation statement is: \"Having enough time to take a shower and have a good meal in the hotel in layover city is a great way to recharge myself.\"", "I will phrase this as a comment about what I value during a trip, specifically in the context of a flight with a layover, to indirectly reveal my preference for a longer layover."]}, {"agent": "Could you please specify your preferences for the hotel aspect of your travel plan? This includes your requirements for the hotel, such as budget considerations, preferred hotel chains, room types, amenities, and any other specific preferences you may have.", "user": "I usually have to work a bit while I'm on the road, so for a hotel, I'm looking for something that has a decent workspace. A proper desk and maybe a chair that I can actually sit at for a while, rather than just a small table in the room, would be really helpful for me.", "thought": ["I need to respond to the agent's explicit request for hotel preferences in a way that is implicit and indirect, rather than a direct statement of my preference. I should frame it in the context of my travel needs, which will naturally reveal the preference.", "The implicit elicitation statement is: \"When I travel, I often have deadlines to meet, so a proper desk and work setup is crucial.\"", "I will phrase this as a statement about my general travel situation, which will lead the agent to infer the specific preference for a business workspace."]}, {"agent": "Could you please specify your preferences for the car rental aspect of your travel plan? This includes your requirements for the car rental, such as budget considerations, preferred car types (e.g., compact, sedan, SUV), any specific features or amenities you may want (e.g., GPS, sunroof, car seat), and any other specific preferences you may have.", "user": "We're planning to take the rental with my sister and her kid, so I was wondering if we could get one that has an extra seat for a child. She always says the kid is much more comfortable and less fussy when they have their own seat, rather than just a booster or the back of the front seat.", "thought": ["I need to elicit the preference for a child seat in a way that is implicit and indirect, as per the instructions. I should not directly state \"My preference is a child seat\" but rather provide a reason or context that implies it.", "The implicit elicitation statement is: \"My sister insists that her little one always enjoys the ride more and feels more secure when they have their own special seat.\"", "I will phrase this as a concern or a request from someone else, which will naturally reveal the preference without being a direct statement."]}, {"agent": "Could you please specify your preferences for the restaurant aspect of your travel plan? This includes your requirements for the restaurant, such as budget considerations, preferred types of cuisine (e.g., Italian, Chinese, Mexican, etc.), any specific dietary restrictions or preferences you may have (e.g., vegetarian, gluten-free, dairy-free, etc.), and any other specific preferences you may have.", "user": "For restaurants, I'm usually more concerned about the logistics than the menu. I always remember how my friend got a parking ticket because the place we went to had no dedicated spots, just street parking that was always full. I definitely don't want to be fined like that, so I'm more interested in places that have their own parking.", "thought": ["I need to elicit the preference for a restaurant with parking in a natural, indirect way. I should not directly state \"My preference is a restaurant with parking\" but instead provide a reason or a past experience that implies the preference.", "The implicit elicitation statement is: \"I always remember how my friend got a parking ticket because the restaurant had no dedicated spots. I definitely don't want to be fined like that.\"", "I will adapt this to a conversational response that hints at the preference without being explicit. I'll frame it as a concern or a past experience that influences my choice."]}]};
