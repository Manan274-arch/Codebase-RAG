import axios from "axios";
export async function sendCheckout(cart: object) { return axios.post("/api/checkout", cart); }
