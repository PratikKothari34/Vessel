'use strict';

/**
 * Retrieval when the archive holds the same memory more than once.
 *
 * A long conversation repeats itself. The user asks the same question every few
 * hundred turns, the character has a line it comes back to, a greeting recurs.
 * Those rows embed to near-identical vectors, so they score near-identically
 * against a query and win adjacent slots -- and the reply is then built on k
 * copies of one memory instead of k memories.
 *
 * The 20,000-turn run put a number on it. Its recall probes are asked once per
 * distance, so the fifth ask met four earlier copies of the question sitting in
 * the archive, every one of them closer to the query than the answer was.
 * Replayed against the run's own database, retrieval returned the question four
 * times over and found the answer in 0 of 40 probes; with duplicates
 * suppressed, the same queries over the same rows found it in 35.
 *
 * The geometry here is chosen rather than hashed. fakeEmbedding hashes the whole
 * string, so two texts are either the same vector or unrelated ones, and "almost
 * the same" -- the entire subject of these tests -- cannot be expressed. Every
 * vector below sits at a stated angle in one plane, so the cosine between any
 * two rows is the cosine of the angle between them, and each expected outcome
 * is arithmetic rather than a guess.
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const backend = require(path.resolve(__dirname, '..', 'helpers', 'backend.js'));
const { EMBED_DIM } = require(path.resolve(__dirname, '..', 'helpers', 'fake-inference.js'));

const ENV = {
  // Archiving is what puts rows in front of retrieval, and the fold is what
  // archives. The shipped default has the summariser off, so it is turned back
  // on here for the same reason memory-fold does.
  SUMMARY_ENABLED: '1',
  VERBATIM_TURNS: '2',
  SUMMARIZE_THRESHOLD: '4',
  MAX_FOLD_TURNS: '24',
  RETRIEVE_K: '4',
  // The score floor is not what these tests are about, and a low one leaves
  // room to place rows more than 14 degrees apart -- the separation below which
  // two vectors are duplicates -- while still clearing it.
  RETRIEVE_MIN_SCORE: '0.05',
};

// A unit vector at `deg` from the query direction, in the plane of the first two
// axes.
function atAngle(deg) {
  const r = (deg * Math.PI) / 180;
  const v = new Array(EMBED_DIM).fill(0);
  v[0] = Math.cos(r);
  v[1] = Math.sin(r);
  return v;
}

// Off that plane entirely, so it can never clear the score floor: what the fold
// embeds for turns a test does not care about, the replies included.
function offPlane(text) {
  const v = new Array(EMBED_DIM).fill(0);
  let h = 0;
  for (let i = 0; i < text.length; i++) h = (h * 31 + text.charCodeAt(i)) >>> 0;
  v[2 + (h % (EMBED_DIM - 2))] = 1;
  return v;
}

const say = (content) => ({ role: 'user', content });

async function exchange(app, convId, characterId, texts) {
  let id = convId;
  for (const t of texts) {
    const out = await app.sse(id ? { conversationId: id, messages: [say(t)] } : { characterId, messages: [say(t)] });
    assert.equal(out.status, 200, `turn "${t}" failed`);
    id = out.meta.conversationId;
  }
  return id;
}

async function waitFor(fn, what, ms = 15000) {
  const deadline = Date.now() + ms;
  for (;;) {
    const v = await fn();
    if (v) return v;
    if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((r) => setTimeout(r, 50));
  }
}

let app;
let hero;

test.before(async () => {
  app = await backend.start({ env: ENV, script: { reply: 'Reply.', summary: 'SUMMARY: so far.' } });
  hero = (await app.post('/characters', { name: 'Mira', persona: 'Terse.' })).json;
});
test.after(async () => { if (app) await app.stop(); });

/**
 * Archive `plan` -- a list of [text, angle] -- in a fresh conversation, then ask
 * `query` and hand back what retrieval chose.
 *
 * The fold embeds an archived turn as "role: content", so the angle map is keyed
 * that way; the query is embedded as itself.
 */
async function recallFor(plan, query) {
  const angles = new Map(plan.map(([text, deg]) => [`user: ${text}`, deg]));
  app.model.script.embedFor = (text) => {
    if (text === query) return atAngle(0);
    const deg = angles.get(text);
    return deg === undefined ? offPlane(text) : atAngle(deg);
  };

  // Two trailing turns push the planted rows out of the verbatim window, so
  // everything under test has to come back through the archive.
  const texts = plan.map(([t]) => t).concat(['filler one', 'filler two']);
  const id = await exchange(app, null, hero.id, texts);
  const want = plan.length * 2;
  await waitFor(async () => {
    const c = (await app.get(`/conversations/${id}`)).json;
    return c.archive.filter((t) => t.hasEmbedding).length >= want ? c : null;
  }, 'the archive to fill');

  const out = await app.sse({ conversationId: id, messages: [say(query)] });
  assert.equal(out.status, 200);
  return out.meta.recalled;
}

test('a line the archive holds four times takes one slot, not all four', async () => {
  const q = 'What is my boat called?';
  const recalled = await recallFor([
    [q, 10], [q, 10], [q, 10], [q, 10],   // the question, asked before
    ['The boat is the Corvane.', 30],       // the answer, further from the query
    ['The harbour is loud.', 50],
    ['It rained all week.', 70],
  ], q);

  assert.equal(recalled.length, 4, 'the budget is still spent in full');
  assert.equal(new Set(recalled.map((r) => r.content)).size, 4, 'on four different memories');
  assert.equal(recalled.filter((r) => r.content === q).length, 1, 'one copy of the question, not four');
  assert.ok(recalled.some((r) => /Corvane/.test(r.content)),
    'the answer reaches the model instead of being crowded out by the question');
});

test('of two copies, the one that matches better is the one kept', async () => {
  // 25 and 34 degrees: 0.906 and 0.829 against the query, 0.988 against each
  // other. One memory, stated twice, the second time less well.
  const recalled = await recallFor([
    ['The lock is set to 3907, I think.', 34],
    ['The lock on my case is set to 3907.', 25],
    ['Gulls again.', 60],
  ], 'What is the combination on my case?');

  const locks = recalled.filter((r) => /3907/.test(r.content));
  assert.equal(locks.length, 1, 'the pair collapses to one');
  assert.equal(locks[0].content, 'The lock on my case is set to 3907.', 'and it is the better copy');
});

test('two memories about one subject are not one memory', async () => {
  // 20 and 40 degrees: 0.940 against each other, under the bar. Turns that share
  // a subject are most of a long conversation, and losing them to duplicate
  // suppression would cost more than the duplicates do.
  const recalled = await recallFor([
    ['My sister is called Thessaly.', 20],
    ['My sister has not spoken to me since the funeral.', 40],
    ['Gulls again.', 70],
  ], 'Tell me about my sister.');

  assert.equal(recalled.filter((r) => /sister/.test(r.content)).length, 2,
    'both survive: near is not the same');
});

test('a duplicate that arrives later and matches worse changes nothing', async () => {
  // The archive is scanned oldest first, so the better copy is already held when
  // the worse one turns up. It must neither replace it nor take a slot of its
  // own, and must not displace the weakest genuine memory either.
  const recalled = await recallFor([
    ['My oldest friend is Cassilis.', 15],
    ['The office safe opens on 5164.', 45],
    ['I grew up in Kalbeck.', 65],
    ['My oldest friend is Cassilis, I said.', 23], // 0.990 against the first
    ['I had a dog called Hesper.', 85],
  ], 'Who is my oldest friend?');

  assert.equal(recalled.length, 4);
  const friend = recalled.filter((r) => /Cassilis/.test(r.content));
  assert.equal(friend.length, 1);
  assert.equal(friend[0].content, 'My oldest friend is Cassilis.', 'the earlier, better copy is held');
  assert.ok(recalled.some((r) => /Hesper/.test(r.content)),
    'the weakest genuine memory gets the slot the duplicate did not take');
});

test('suppression cannot pad the answer out to k', async () => {
  // Only one memory clears the floor, stated twice. Neither duplicate handling
  // nor the cutoff may invent a second.
  const recalled = await recallFor([
    ['My editor is Nembrot.', 20],
    ['My editor is Nembrot, as I said.', 27], // 0.993 -- the same memory
    ['Gulls again.', 89],                      // 0.017, under the floor
  ], 'Who is my editor?');

  assert.equal(recalled.length, 1, 'one memory in, one memory out');
  assert.equal(recalled[0].content, 'My editor is Nembrot.');
});
