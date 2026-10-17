// Keep the phone's session cache in line with the laptop, which is the source
// of truth for results and labels (labels may also come from the laptop's
// browser test bench or another device). Runs when Home / Stats open.
import * as api from './api';
import { getSessions, saveSession, summarizeShots, keepOnlySessions } from '../utils/storage';

let running = null;


/** Pull new or changed sessions from the laptop. Resolves to true if anything changed.
 *  Silently does nothing when the laptop is offline. */
export function syncSessions() {
  if (running) return running;   // one sync at a time
  running = (async () => {
    let changed = false;
    try {
      const remote = await api.sessions();
      // The laptop lists exactly this account's sessions: anything else cached
      // here (deleted on the laptop, or left from another account) goes.
      if (await keepOnlySessions(remote.map(r => r.id))) changed = true;
      const local = new Map((await getSessions()).map(s => [s.id, s]));
      for (const r of remote) {
        const mine = local.get(r.id);
        const sum = mine?.result ? summarizeShots(mine.result.shots || []) : null;
        const same = sum && sum.total === r.total && sum.labeled === r.labeled && sum.made === r.made;
        if (same) continue;
        try {
          const job = await api.job(r.id);
          if (!job.result) continue;
          await saveSession(r.id, job.result, { localUri: mine?.localUri, timestamp: r.created });
          changed = true;
        } catch (_) { /* skip this one; try again next sync */ }
      }
    } catch (_) {
      // laptop offline / not signed in – keep the local cache
    } finally {
      running = null;
    }
    return changed;
  })();
  return running;
}
