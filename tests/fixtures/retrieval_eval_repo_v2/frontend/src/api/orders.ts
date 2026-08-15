import axios from "axios";
export async function submitOrder(payload: object) { return axios.post("/api/orders", payload); }
export async function fetchOrder(orderId: string) { return axios.get(`/api/orders/${orderId}`); }
export async function requestCancellation(orderId: string) { return axios.delete(`/api/orders/${orderId}`); }
