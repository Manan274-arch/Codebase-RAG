import axios from "axios";
export async function submitRefund(payload: object) { return axios.post("/api/refunds", payload); }
