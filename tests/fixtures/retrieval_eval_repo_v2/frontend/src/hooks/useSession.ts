import { renewSession } from "../api/auth";
export async function restoreSession(refreshToken: string) { return renewSession(refreshToken); }
