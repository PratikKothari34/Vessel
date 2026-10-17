/**
 * Does a chat stream's result still belong on screen?
 *
 * Kept apart from the component for the same reason as settings-state.mjs: this
 * is where the decision is, and where the bug was. A reply is streamed for one
 * conversation, but it arrives whenever the model finishes -- and the user is
 * free to click another scenario in the rail while it is still generating.
 * Aborting does not help: `streamChat`'s abort sets a flag and fires a
 * best-effort `cancel_chat`, it does NOT detach `channel.onmessage`, so a
 * `done` or `error` already in flight still runs the handlers the original send
 * installed, with the original conversation id captured in their closure.
 *
 * Three things then went wrong at once, all observed:
 *
 * - `onDone` appended the finished reply to whatever conversation was on screen
 *   now, so conversation A's text landed in conversation B's transcript.
 * - It then reloaded the OLD id, and because that reload resolved after the
 *   switch's own reload, conversation A's turns overwrote the visible B.
 * - `onError` rolled back to the `messages` array captured at send time, which
 *   is the other conversation's history -- it put A's turns on screen under B's
 *   heading, which is the same class of leak as the append.
 *
 * `stopAndReload` already had this guard and the comment explaining why
 * ("so a reload never clobbers a conversation switched to since"). The
 * completion paths simply never got it. Expressed here as plain data in and a
 * boolean out, so the suite can pin every ordering without a DOM.
 *
 * .mjs rather than .js because app/package.json declares no type, so Node reads
 * a .js file here as CommonJS and the tests could not import it.
 */

/**
 * Is `streamConvId` -- the conversation a stream was started for -- still the
 * one the user is looking at (`currentConvId`)?
 *
 * Both ids are compared as they are stored: a string id, or `null` for a
 * scenario that has not been minted server-side yet. A brand-new conversation
 * starts as `null` and is assigned an id mid-stream via `onMeta`, so the send
 * path updates its captured id when that arrives; by the time a result lands,
 * a stream that still holds `null` was never adopted and its result cannot be
 * placed. Returning false there is deliberate: the alternative is writing a
 * finished reply into whichever scenario happens to be open.
 *
 * Total over its arguments. It is called from a stream callback during an
 * unmount race, where a throw escapes to window.onerror rather than to any
 * component's error state, so a non-string id must compare false, not throw.
 */
export function ownsStream(streamConvId, currentConvId) {
  if (typeof streamConvId !== 'string' || !streamConvId) return false;
  if (typeof currentConvId !== 'string' || !currentConvId) return false;
  return streamConvId === currentConvId;
}
