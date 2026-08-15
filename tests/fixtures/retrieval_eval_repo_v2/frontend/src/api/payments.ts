import axios from "axios";
export async function authorizeCard(payload: object) { return axios.post("/api/payments/authorize", payload); }
export async function previewPayment(payload: object) { return { status: "preview", payload }; }
