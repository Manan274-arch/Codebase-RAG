import axios from "axios";
export async function trackShipment(orderId: string) { return axios.get(`/api/shipping/${orderId}`); }
