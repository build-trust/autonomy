export interface Message {
  role: "user" | "assistant";
  content: string;
}

export interface StreamResponse {
  messages?: Array<{
    content?: {
      text?: string;
    };
  }>;
}
