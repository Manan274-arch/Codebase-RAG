import axios from "axios";
export async function signIn(email: string, password: string) { return axios.post("/api/auth/login", { email, password }); }
export async function renewSession(refreshToken: string) { return axios.post("/api/auth/refresh", { refresh_token: refreshToken }); }
