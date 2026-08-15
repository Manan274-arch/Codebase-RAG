import axios from "axios";
export async function loadCurrentProfile() { return axios.get("/api/profile"); }
export async function saveProfile(changes: object) { return axios.patch("/api/profile", changes); }
