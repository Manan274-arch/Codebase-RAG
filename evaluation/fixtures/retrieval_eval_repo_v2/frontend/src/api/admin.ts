import axios from "axios";
export async function approveRefund(refundId: string) { return axios.post(`/api/admin/refunds/${refundId}/approve`, {}); }
