/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */
// ---------------------------------------------------------------------------
// sentences.js -- decoding examples that drive the interactive explorer.
//
// Schema (timing-ready):
//   {
//     id,                     // unique string
//     source: 'MEG' | 'EEG',
//     subject: 'best' | 'median' | 'worst',
//     version: 'v1' | 'v2',
//     truth:   '...',         // the sentence the participant read
//     typed:   '...' | null,  // what they actually typed (null if unknown)
//     decoded: '...',         // what Brain2Qwerty decoded from brain activity
//     keystrokes?: [{ char, tMs }]   // OPTIONAL per-key timing -> enables true
//                                     // timed playback (drop v2 data in here)
//   }
//
// v1 examples below are transcribed from Lévy et al. (Brain2Qwerty, accepted at
// Nature Neuroscience): Table 1 (best-decoded MEG/EEG) and Figure 3 (best /
// median / worst MEG subjects). v2 slots are intentionally empty -- they will be
// filled with the v2 sentence examples (with per-keystroke timing) once shared.
// ---------------------------------------------------------------------------

export const SENTENCES = [
  // ---------------- MEG · best subject (v1) ----------------
  {
    id: 'meg-best-1', source: 'MEG', subject: 'best', version: 'v1',
    truth: 'el beneficio supera los riesgos',
    typed: 'ek benefucui syoera kis ruesgis',
    decoded: 'el beneficio supera los riesgos',
    note: 'The language model fixes the participant\u2019s own typos.',
  },
  {
    id: 'meg-best-2', source: 'MEG', subject: 'best', version: 'v1',
    truth: 'la silla ocasiona las lesiones',
    typed: 'la silla ocasioma las lesiomes',
    decoded: 'la silla ocasiona las lesiones',
  },
  {
    id: 'meg-best-3', source: 'MEG', subject: 'best', version: 'v1',
    truth: 'las teorias reducen los numeros',
    typed: 'las teorias reducen los numeros',
    decoded: 'las teorias reducen los numeros',
  },

  // ---------------- MEG · median subject (v1, Fig. 3) ----------------
  {
    id: 'meg-median-1', source: 'MEG', subject: 'median', version: 'v1',
    truth: 'las teorias reducen los numeros',
    typed: null,
    decoded: 'las teorias exigen los hombros',
  },
  {
    id: 'meg-median-2', source: 'MEG', subject: 'median', version: 'v1',
    truth: 'la estadistica sigue la distribucion',
    typed: null,
    decoded: 'stamistosa sigue la distribucion',
  },

  // ---------------- MEG · worst subject (v1, Fig. 3) ----------------
  {
    id: 'meg-worst-1', source: 'MEG', subject: 'worst', version: 'v1',
    truth: 'las teorias reducen los numeros',
    typed: null,
    decoded: 'las rancias revisen los numerad',
  },
  {
    id: 'meg-worst-2', source: 'MEG', subject: 'worst', version: 'v1',
    truth: 'la estadistica sigue la distribucion',
    typed: null,
    decoded: 'la estadistica figura de petrilla lo',
  },

  // ---------------- EEG · best subject (v1, Table 1) ----------------
  {
    id: 'eeg-best-1', source: 'EEG', subject: 'best', version: 'v1',
    truth: 'la ciencia de la idea rompe la vision',
    typed: 'la ciencia de la idea rompe la bision',
    decoded: 'la ciencia de la idea las mas de esos',
  },
  {
    id: 'eeg-best-2', source: 'EEG', subject: 'best', version: 'v1',
    truth: 'el procesador ejecuta la instruccion',
    typed: 'ordenador ejecuta la instruccion',
    decoded: 'las corrida perita la instruccion',
  },
  {
    id: 'eeg-best-3', source: 'EEG', subject: 'best', version: 'v1',
    truth: 'la presencia de los tipos impone los retos',
    typed: 'la presencia de los tipos impone los retos',
    decoded: 'la declarada de los celos eran a los actos',
  },

  // ---------------- v2 slots (to be provided, with timing) ----------------
  // Example shape for when v2 data arrives:
  // {
  //   id: 'meg-best-v2-1', source: 'MEG', subject: 'best', version: 'v2',
  //   truth: '...', typed: '...', decoded: '...',
  //   keystrokes: [{ char: 'l', tMs: 0 }, { char: 'a', tMs: 160 }, ...],
  // },
];

// Ablation walkthrough (MEG, Table 2): how each stage cleans up one sentence.
export const ABLATION = {
  truth: 'el beneficio supera los riesgos',
  stages: [
    { label: 'Typed', text: 'el bemeficio supera los riesfos' },
    { label: 'Conv', text: 'el gefedisio suiera noa riestii' },
    { label: 'Conv + Trans', text: 'el geneficon cupera los riesgoo' },
    { label: 'Brain2Qwerty', text: 'el beneficio supera los riesgos', final: true },
  ],
};
