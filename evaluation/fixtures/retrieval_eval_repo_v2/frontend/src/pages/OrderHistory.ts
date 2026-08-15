import { fetchOrder } from "../api/orders";
export async function openOrder(orderId: string) { return (await fetchOrder(orderId)).data; }
