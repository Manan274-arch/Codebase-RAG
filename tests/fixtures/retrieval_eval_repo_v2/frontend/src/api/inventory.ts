import axios from "axios";
export async function loadAvailability(sku: string) { return axios.get(`/api/inventory/${sku}`); }
export async function holdInventory(sku: string, quantity: number) { return axios.patch(`/api/inventory/${sku}/reserve`, { quantity }); }
