import { signIn } from "../api/auth";
export async function submitCredentials(email: string, password: string) { return signIn(email, password); }
