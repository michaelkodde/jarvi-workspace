// dashboard/chat.js
/* global Alpine, marked, DOMPurify */

function chatApp() {
    return {
        messages: [],
        inputMessage: '',
        isProcessing: false,
        hitlTicket: null,
        ws: null,
        conversationId: null,
        wsStatus: 'disconnected',
        wsStatusText: 'Disconnected',
        reconnectAttempts: 0,
        maxReconnectAttempts: 10,

        init() {
            // Generate or restore conversation ID
            this.conversationId =
                localStorage.getItem('jarvi_conversation_id') ||
                crypto.randomUUID().slice(0, 12);
            localStorage.setItem('jarvi_conversation_id', this.conversationId);
            this.connectWebSocket();
        },

        connectWebSocket() {
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                this.wsStatus = 'disconnected';
                this.wsStatusText = 'Connection failed - refresh page';
                return;
            }

            this.wsStatus = 'connecting';
            this.wsStatusText = 'Connecting...';

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const hostname = window.location.hostname;
            const port = window.location.port ? `:${window.location.port}` : '';
            const wsUrl = `${protocol}//${hostname}${port}/ws/chat/${this.conversationId}`;

            try {
                this.ws = new WebSocket(wsUrl);
            } catch (e) {
                console.error('Failed to create WebSocket:', e);
                this.wsStatus = 'disconnected';
                this.wsStatusText = 'Connection error';
                return;
            }

            this.ws.onopen = () => {
                this.wsStatus = 'connected';
                this.wsStatusText = 'Connected';
                this.reconnectAttempts = 0;
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWSEvent(data);
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onclose = () => {
                this.wsStatus = 'disconnected';
                this.wsStatusText = 'Disconnected';
                this.reconnectAttempts++;
                // Exponential backoff: 3s, 6s, 12s, 24s, max 30s
                const delay = Math.min(3000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
                setTimeout(() => this.connectWebSocket(), delay);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        },

        handleWSEvent(data) {
            if (!data || !data.type || !data.payload) return;

            switch (data.type) {
                case 'text_delta':
                    this.addMessage('assistant', data.payload.content || '');
                    break;
                case 'status_update':
                    this.addMessage(
                        'system',
                        `Status: ${data.payload.status || 'unknown'} (${data.payload.step || ''})`
                    );
                    break;
                case 'hitl_required':
                    this.hitlTicket = {
                        ticketId: data.payload.ticket_id,
                        tier: data.payload.tier,
                        preview: data.payload.preview || '',
                    };
                    this.addMessage('system', 'Job requires your approval');
                    break;
                case 'completed':
                    this.hitlTicket = null;
                    this.isProcessing = false;
                    const rawContent = data.payload.result_preview || '';
                    const markdown = DOMPurify.sanitize(marked.parse(rawContent));
                    this.addMessage('assistant', markdown, 'markdown');
                    break;
                case 'ack':
                    // Acknowledgment - no UI action needed
                    break;
                case 'error':
                    this.isProcessing = false;
                    this.addMessage('error', data.payload.message || 'Unknown error');
                    break;
                default:
                    console.warn('Unknown WebSocket event type:', data.type);
            }
            this.$nextTick(() => {
                if (this.$refs.messages) {
                    this.$refs.messages.scrollTop = this.$refs.messages.scrollHeight;
                }
            });
        },

        sendMessage() {
            const content = this.inputMessage.trim();
            if (!content || this.isProcessing || this.wsStatus !== 'connected') return;

            this.addMessage('user', content);
            try {
                this.ws.send(JSON.stringify({
                    type: 'user_message',
                    conversation_id: this.conversationId,
                    content: content,
                }));
            } catch (e) {
                console.error('Failed to send message:', e);
                this.addMessage('error', 'Failed to send message');
                return;
            }
            this.inputMessage = '';
            this.isProcessing = true;
        },

        approveHitl() {
            if (!this.hitlTicket || this.wsStatus !== 'connected') return;
            try {
                this.ws.send(JSON.stringify({
                    type: 'approve_hitl',
                    ticket_id: this.hitlTicket.ticketId,
                    action: 'approve',
                }));
                this.hitlTicket = null;
            } catch (e) {
                console.error('Failed to send approval:', e);
            }
        },

        rejectHitl() {
            if (!this.hitlTicket || this.wsStatus !== 'connected') return;
            try {
                this.ws.send(JSON.stringify({
                    type: 'approve_hitl',
                    ticket_id: this.hitlTicket.ticketId,
                    action: 'reject',
                }));
                this.hitlTicket = null;
            } catch (e) {
                console.error('Failed to send rejection:', e);
            }
        },

        addMessage(role, content, type = 'text') {
            this.messages.push({
                id: Date.now() + Math.random(),
                role,
                content,
                type,
            });
        },
    };
}