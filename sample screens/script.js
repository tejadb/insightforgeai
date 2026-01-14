// ============================================================================
// Configuration
// ============================================================================

// API URL - Update this to your Render API URL after deployment
const API_URL = window.API_URL || 'http://127.0.0.1:8008';

// Get user_id from localStorage
function getUserId() {
    const userId = localStorage.getItem('user_id');
    if (!userId) {
        window.location.href = 'auth.html';
        return null;
    }
    return userId;
}

// ============================================================================
// UI State Management
// ============================================================================

let selectedDocIds = []; // Array of selected document IDs
let documents = {}; // Map of doc_id -> document data
let insights = {}; // Map of insight_id -> insight data
let pollingIntervals = {}; // Map of doc_id/insight_id -> interval ID

// ============================================================================
// Panel Toggle Functions
// ============================================================================

let isLeftCollapsed = false;
let isRightCollapsed = false;

function toggleLeftPanel() {
    isLeftCollapsed = !isLeftCollapsed;
    updateWorkspaceLayout();
}

function toggleRightPanel() {
    isRightCollapsed = !isRightCollapsed;
    updateWorkspaceLayout();
}

function updateWorkspaceLayout() {
    const workspace = document.getElementById('workspace');
    const leftBtn = document.getElementById('leftCollapsedBtn');
    const rightBtn = document.getElementById('rightCollapsedBtn');
    
    workspace.classList.remove('left-collapsed', 'right-collapsed', 'both-collapsed');
    
    if (isLeftCollapsed && isRightCollapsed) {
        workspace.classList.add('both-collapsed');
    } else if (isLeftCollapsed) {
        workspace.classList.add('left-collapsed');
    } else if (isRightCollapsed) {
        workspace.classList.add('right-collapsed');
    }

    leftBtn.style.display = isLeftCollapsed ? 'flex' : 'none';
    rightBtn.style.display = isRightCollapsed ? 'flex' : 'none';
}

// ============================================================================
// Document Management
// ============================================================================

function toggleSource(element) {
    const docId = element.dataset.docId;
    if (!docId) return;
    
    const isActive = element.classList.contains('active');
    if (isActive) {
        element.classList.remove('active');
        selectedDocIds = selectedDocIds.filter(id => id !== docId);
    } else {
        // Only allow selection if document is completed
        const doc = documents[docId];
        if (doc && doc.status === 'completed') {
            element.classList.add('active');
            selectedDocIds.push(docId);
        }
    }
    updateSourceCount();
    updateChatInputState();
}

function updateSourceCount() {
    const activeSources = document.querySelectorAll('.source-item.active').length;
    const countEl = document.getElementById('sourceCount');
    if (countEl) {
        countEl.textContent = activeSources;
    }
}

function updateChatInputState() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.querySelector('.send-btn');
    const hasSelected = selectedDocIds.length > 0;
    
    if (chatInput) {
        chatInput.disabled = !hasSelected;
        chatInput.placeholder = hasSelected 
            ? `Type your question here... (${selectedDocIds.length} document${selectedDocIds.length > 1 ? 's' : ''} selected)`
            : 'Select documents to start chatting...';
    }
    if (sendBtn) {
        sendBtn.disabled = !hasSelected;
    }
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return 'fa-file-pdf';
    if (ext === 'pptx' || ext === 'ppt') return 'fa-file-powerpoint';
    if (ext === 'docx' || ext === 'doc') return 'fa-file-word';
    if (ext === 'xlsx' || ext === 'xls') return 'fa-file-excel';
    return 'fa-file';
}

function getFileIconColor(filename) {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (ext === 'pdf') return '#ef4444';
    if (ext === 'pptx' || ext === 'ppt') return '#f97316';
    if (ext === 'docx' || ext === 'doc') return '#2b579a';
    if (ext === 'xlsx' || ext === 'xls') return '#16a34a';
    return '#5f6368';
}

function renderDocumentStatus(doc) {
    const status = doc.status;
    if (status === 'pending' || status === 'processing') {
        return '<i class="fa-solid fa-spinner fa-spin" style="color: #6366f1;"></i>';
    } else if (status === 'completed') {
        return '<i class="fa-solid fa-check" style="color: #10b981;"></i>';
    } else if (status === 'error') {
        return '<i class="fa-solid fa-times" style="color: #ef4444;"></i>';
    }
    return '';
}

function renderDocumentItem(doc) {
    const isSelected = selectedDocIds.includes(doc.id);
    const iconClass = getFileIcon(doc.title);
    const iconColor = getFileIconColor(doc.title);
    const statusIcon = renderDocumentStatus(doc);
    const canSelect = doc.status === 'completed';
    
    return `
        <div class="source-item ${isSelected ? 'active' : ''} ${canSelect ? '' : 'disabled'}" 
             data-doc-id="${doc.id}" 
             onclick="${canSelect ? 'toggleSource(this)' : ''}">
            <i class="fa-regular ${iconClass} source-icon" style="color: ${iconColor};"></i>
            <div class="source-info">
                <span class="source-name" style="${doc.status !== 'completed' ? 'opacity: 0.6;' : ''}">${escapeHtml(doc.title)}</span>
                <span class="source-meta">
                    ${doc.status === 'pending' ? 'Queued...' : ''}
                    ${doc.status === 'processing' ? 'Processing...' : ''}
                    ${doc.status === 'completed' ? 'Ready' : ''}
                    ${doc.status === 'error' ? 'Error' : ''}
                </span>
            </div>
            <div class="selection-indicator">
                ${statusIcon}
            </div>
        </div>
    `;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function loadDocuments() {
    const userId = getUserId();
    if (!userId) return;

    try {
        const response = await fetch(`${API_URL}/documents?user_id=${userId}`);
        if (!response.ok) throw new Error('Failed to load documents');
        
        const data = await response.json();
        documents = {};
        const sourcesList = document.getElementById('sourcesList');
        if (!sourcesList) return;

        sourcesList.innerHTML = '';

        if (data.documents && data.documents.length > 0) {
            data.documents.forEach(doc => {
                documents[doc.id] = doc;
                sourcesList.innerHTML += renderDocumentItem(doc);
            });
            
            // Start polling for pending/processing documents
            data.documents.forEach(doc => {
                if (doc.status === 'pending' || doc.status === 'processing') {
                    startPollingDocument(doc.id);
                }
            });
        } else {
            sourcesList.innerHTML = '<div style="text-align: center; color: #5f6368; padding: 20px;">No documents yet. Upload one to get started!</div>';
        }

        updateSourceCount();
    } catch (error) {
        console.error('Error loading documents:', error);
        showNotification('Failed to load documents', 'error');
    }
}

function startPollingDocument(docId) {
    // Clear existing interval if any
    if (pollingIntervals[docId]) {
        clearInterval(pollingIntervals[docId]);
    }

    pollingIntervals[docId] = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/documents/${docId}`);
            if (!response.ok) throw new Error('Failed to check document status');
            
            const doc = await response.json();
            documents[docId] = doc;

            // Update UI
            const docElement = document.querySelector(`[data-doc-id="${docId}"]`);
            if (docElement) {
                docElement.outerHTML = renderDocumentItem(doc);
            }

            // Stop polling if completed or error
            if (doc.status === 'completed' || doc.status === 'error') {
                clearInterval(pollingIntervals[docId]);
                delete pollingIntervals[docId];
            }
        } catch (error) {
            console.error(`Error polling document ${docId}:`, error);
            clearInterval(pollingIntervals[docId]);
            delete pollingIntervals[docId];
        }
    }, 3000); // Poll every 3 seconds
}

async function uploadDocument(file) {
    const userId = getUserId();
    if (!userId) return;

    const sourcesList = document.getElementById('sourcesList');
    if (!sourcesList) return;

    // Create temporary UI element
    const tempId = 'temp-' + Date.now();
    const tempDoc = {
        id: tempId,
        title: file.name,
        status: 'pending'
    };
    documents[tempId] = tempDoc;

    const tempElement = document.createElement('div');
    tempElement.innerHTML = renderDocumentItem(tempDoc);
    sourcesList.insertBefore(tempElement.firstElementChild, sourcesList.firstChild);

    try {
        const formData = new FormData();
        formData.append('user_id', userId);
        formData.append('file', file);

        const response = await fetch(`${API_URL}/documents`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }

        const data = await response.json();
        
        // Remove temp element
        const tempEl = document.querySelector(`[data-doc-id="${tempId}"]`);
        if (tempEl) tempEl.remove();

        // Add real document
        documents[data.doc_id] = {
            id: data.doc_id,
            title: data.title,
            status: data.status
        };
        sourcesList.innerHTML = renderDocumentItem(documents[data.doc_id]) + sourcesList.innerHTML;

        // Start polling
        startPollingDocument(data.doc_id);

        showNotification('Document uploaded successfully', 'success');
    } catch (error) {
        console.error('Error uploading document:', error);
        
        // Update temp element to show error
        const tempEl = document.querySelector(`[data-doc-id="${tempId}"]`);
        if (tempEl) {
            tempEl.querySelector('.source-meta').textContent = 'Upload failed';
            tempEl.querySelector('.selection-indicator').innerHTML = '<i class="fa-solid fa-times" style="color: #ef4444;"></i>';
        }
        
        delete documents[tempId];
        showNotification(error.message || 'Upload failed', 'error');
    }
}

// ============================================================================
// Insight Management
// ============================================================================

const modalTitles = {
    'summary': 'Generate Document Summary',
    'qna': 'Create Practice Q&A',
    'interview_tips': 'Extract Interview Tips',
    'tips': 'Extract Interview Tips',
    'slides': 'Customize Slide Deck'
};

const defaultInstructions = {
    'summary': 'Focus on the key findings and financial metrics.',
    'qna': 'Create a mix of technical and behavioral questions.',
    'tips': 'Highlight areas where I can demonstrate leadership.',
    'interview_tips': 'Highlight areas where I can demonstrate leadership.',
    'slides': 'Create a 10-slide deck for a managerial audience.'
};

let currentAction = '';

function openModal(actionType) {
    // Check if documents are selected
    if (selectedDocIds.length === 0) {
        showNotification('Please select at least one document first', 'error');
        return;
    }

    currentAction = actionType;
    const modal = document.getElementById('actionModal');
    const title = document.getElementById('modalTitle');
    const instructions = document.getElementById('modalInstructions');

    title.textContent = modalTitles[actionType] || 'Generate Insight';
    instructions.placeholder = defaultInstructions[actionType] || 'Add specific instructions...';
    instructions.value = '';

    modal.classList.add('active');
}

function closeModal() {
    const modal = document.getElementById('actionModal');
    modal.classList.remove('active');
}

async function startGeneration() {
    const userId = getUserId();
    if (!userId) return;

    const instructions = document.getElementById('modalInstructions').value.trim();
    closeModal();

    if (selectedDocIds.length === 0) {
        showNotification('Please select at least one document', 'error');
        return;
    }

    // Map action type
    let actionType = currentAction;
    if (currentAction === 'tips') {
        actionType = 'interview_tips';
    }

    const historyList = document.getElementById('historyList');
    if (!historyList) return;

    // Create temporary history item
    const tempId = 'temp-insight-' + Date.now();
    const item = document.createElement('div');
    item.className = 'history-item generating';
    item.dataset.insightId = tempId;

    let iconClass = 'fa-file-lines';
    if (currentAction === 'qna') iconClass = 'fa-clipboard-question';
    if (currentAction === 'tips' || currentAction === 'interview_tips') iconClass = 'fa-lightbulb';
    if (currentAction === 'slides') iconClass = 'fa-presentation-screen';

    item.innerHTML = `
        <div class="history-icon"><i class="fa-solid ${iconClass}"></i></div>
        <div class="history-content">
            <span class="history-title">${modalTitles[currentAction] || 'Generating...'}</span>
            <span class="history-date">Generating...</span>
        </div>
        <div class="history-loader"><i class="fa-solid fa-spinner fa-spin"></i></div>
    `;

    historyList.insertBefore(item, historyList.firstChild);

    try {
        const response = await fetch(`${API_URL}/insights`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId,
                action_type: actionType,
                document_ids: selectedDocIds,
                instructions: instructions,
                title: null
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create insight');
        }

        const data = await response.json();
        
        // Remove temp item
        item.remove();

        // Add real insight
        insights[data.insight_id] = {
            id: data.insight_id,
            action_type: actionType,
            status: data.status,
            title: modalTitles[currentAction]
        };

        // Render insight in history
        renderInsightItem(data.insight_id, insights[data.insight_id]);

        // Start polling
        startPollingInsight(data.insight_id);

        showNotification('Insight generation started', 'success');
    } catch (error) {
        console.error('Error creating insight:', error);
        item.remove();
        showNotification(error.message || 'Failed to create insight', 'error');
    }
}

function renderInsightItem(insightId, insight) {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;

    const item = document.createElement('div');
    item.className = `history-item ${insight.status === 'completed' ? 'completed' : 'generating'}`;
    item.dataset.insightId = insightId;

    let iconClass = 'fa-file-lines';
    if (insight.action_type === 'qna') iconClass = 'fa-clipboard-question';
    if (insight.action_type === 'interview_tips' || insight.action_type === 'tips') iconClass = 'fa-lightbulb';
    if (insight.action_type === 'slides') iconClass = 'fa-presentation-screen';

    const statusText = insight.status === 'completed' ? 'Just now' : 
                      insight.status === 'error' ? 'Error' : 'Generating...';
    const statusIcon = insight.status === 'completed' ? 
                      '<i class="fa-solid fa-check" style="color: #10b981;"></i>' :
                      insight.status === 'error' ?
                      '<i class="fa-solid fa-times" style="color: #ef4444;"></i>' :
                      '<i class="fa-solid fa-spinner fa-spin"></i>';

    item.innerHTML = `
        <div class="history-icon"><i class="fa-solid ${iconClass}"></i></div>
        <div class="history-content">
            <span class="history-title">${insight.title || modalTitles[insight.action_type] || 'Insight'}</span>
            <span class="history-date">${statusText}</span>
        </div>
        <div class="history-loader">${statusIcon}</div>
    `;

    historyList.insertBefore(item, historyList.firstChild);
}

function startPollingInsight(insightId) {
    // Clear existing interval if any
    if (pollingIntervals[insightId]) {
        clearInterval(pollingIntervals[insightId]);
    }

    pollingIntervals[insightId] = setInterval(async () => {
        try {
            const response = await fetch(`${API_URL}/insights/${insightId}`);
            if (!response.ok) throw new Error('Failed to check insight status');
            
            const insight = await response.json();
            insights[insightId] = insight;

            // Update UI
            const insightElement = document.querySelector(`[data-insight-id="${insightId}"]`);
            if (insightElement) {
                insightElement.outerHTML = '';
                renderInsightItem(insightId, insight);
            }

            // Stop polling if completed or error
            if (insight.status === 'completed' || insight.status === 'error') {
                clearInterval(pollingIntervals[insightId]);
                delete pollingIntervals[insightId];
            }
        } catch (error) {
            console.error(`Error polling insight ${insightId}:`, error);
            clearInterval(pollingIntervals[insightId]);
            delete pollingIntervals[insightId];
        }
    }, 2000); // Poll every 2 seconds
}

async function loadInsights() {
    const userId = getUserId();
    if (!userId) return;

    try {
        const response = await fetch(`${API_URL}/insights?user_id=${userId}`);
        if (!response.ok) throw new Error('Failed to load insights');
        
        const data = await response.json();
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        // Clear existing items (except temp ones)
        historyList.innerHTML = '';

        if (data.insights && data.insights.length > 0) {
            data.insights.forEach(insight => {
                insights[insight.id] = insight;
                renderInsightItem(insight.id, insight);
                
                // Start polling if not completed
                if (insight.status !== 'completed' && insight.status !== 'error') {
                    startPollingInsight(insight.id);
                }
            });
        }
    } catch (error) {
        console.error('Error loading insights:', error);
    }
}

// ============================================================================
// Chat Management
// ============================================================================

async function sendMessage() {
    const userId = getUserId();
    if (!userId) return;

    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    
    if (!text) return;
    if (selectedDocIds.length === 0) {
        showNotification('Please select at least one document', 'error');
        return;
    }

    const chatArea = document.getElementById('chatArea');
    if (!chatArea) return;

    // Remove empty state
    const emptyState = document.querySelector('.empty-state');
    if (emptyState) emptyState.style.display = 'none';

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.style.cssText = 'background: #eff6ff; padding: 12px 16px; border-radius: 12px 12px 0 12px; align-self: flex-end; max-width: 80%; margin-bottom: 12px; font-size: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);';
    userMsg.textContent = text;
    chatArea.appendChild(userMsg);

    input.value = '';
    input.disabled = true;
    updateChatInputState();

    // Add loading message
    const loadingMsg = document.createElement('div');
    loadingMsg.style.cssText = 'background: white; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 12px 12px 12px 0; align-self: flex-start; max-width: 80%; margin-bottom: 12px; font-size: 14px; display: flex; align-items: center; gap: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);';
    loadingMsg.innerHTML = `
        <i class="fa-solid fa-wand-magic-sparkles" style="color: #6366f1;"></i>
        <span style="color: #6b7280;">Thinking...</span>
    `;
    chatArea.appendChild(loadingMsg);
    chatArea.scrollTop = chatArea.scrollHeight;

    try {
        const response = await fetch(`${API_URL}/chat/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId,
                message: text,
                document_ids: selectedDocIds
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Chat failed');
        }

        const data = await response.json();
        
        // Remove loading message
        loadingMsg.remove();

        // Add AI response
        const aiMsg = document.createElement('div');
        aiMsg.style.cssText = 'background: white; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 12px 12px 12px 0; align-self: flex-start; max-width: 80%; margin-bottom: 12px; font-size: 14px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.05); animation: fadeIn 0.3s ease;';
        
        aiMsg.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-weight: 500; color: #6366f1;">
                <i class="fa-solid fa-wand-magic-sparkles"></i> InsightForge
            </div>
            <div>${escapeHtml(data.content || 'No response')}</div>
        `;
        
        chatArea.appendChild(aiMsg);
        chatArea.scrollTop = chatArea.scrollHeight;
    } catch (error) {
        console.error('Error sending message:', error);
        loadingMsg.remove();
        
        const errorMsg = document.createElement('div');
        errorMsg.style.cssText = 'background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; padding: 12px 16px; border-radius: 12px; align-self: flex-start; max-width: 80%; margin-bottom: 12px; font-size: 14px;';
        errorMsg.textContent = `Error: ${error.message}`;
        chatArea.appendChild(errorMsg);
        chatArea.scrollTop = chatArea.scrollHeight;
        
        showNotification(error.message || 'Chat failed', 'error');
    } finally {
        input.disabled = false;
        updateChatInputState();
    }
}

async function loadChatHistory() {
    const userId = getUserId();
    if (!userId) return;

    try {
        const response = await fetch(`${API_URL}/chat/history?user_id=${userId}&limit=200`);
        if (!response.ok) throw new Error('Failed to load chat history');
        
        const data = await response.json();
        const chatArea = document.getElementById('chatArea');
        if (!chatArea) return;

        // Clear existing messages
        chatArea.innerHTML = '';

        if (data.messages && data.messages.length > 0) {
            data.messages.forEach(msg => {
                const msgDiv = document.createElement('div');
                const isUser = msg.role === 'user';
                
                msgDiv.style.cssText = isUser
                    ? 'background: #eff6ff; padding: 12px 16px; border-radius: 12px 12px 0 12px; align-self: flex-end; max-width: 80%; margin-bottom: 12px; font-size: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);'
                    : 'background: white; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 12px 12px 12px 0; align-self: flex-start; max-width: 80%; margin-bottom: 12px; font-size: 14px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.05);';
                
                if (isUser) {
                    msgDiv.textContent = msg.content || '';
                } else {
                    msgDiv.innerHTML = `
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-weight: 500; color: #6366f1;">
                            <i class="fa-solid fa-wand-magic-sparkles"></i> InsightForge
                        </div>
                        <div>${escapeHtml(msg.content || '')}</div>
                    `;
                }
                
                chatArea.appendChild(msgDiv);
            });
            
            chatArea.scrollTop = chatArea.scrollHeight;
        } else {
            // Show empty state
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';
            emptyState.innerHTML = `
                <div class="sparkle-icon">✨</div>
                <h2>Ask anything about your documents</h2>
                <p>Select a source to get started.</p>
            `;
            chatArea.appendChild(emptyState);
        }
    } catch (error) {
        console.error('Error loading chat history:', error);
    }
}

// ============================================================================
// Notification System
// ============================================================================

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-size: 14px;
        font-weight: 500;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: slideIn 0.3s ease;
    `;
    
    if (type === 'success') {
        notification.style.background = '#10b981';
    } else if (type === 'error') {
        notification.style.background = '#ef4444';
    } else {
        notification.style.background = '#6366f1';
    }
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// ============================================================================
// Event Listeners
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    if (!getUserId()) {
        return; // Will redirect to auth.html
    }

    // Load initial data
    loadDocuments();
    loadInsights();
    loadChatHistory();

    // Upload button
    const addSourceBtn = document.querySelector('.add-source-btn');
    if (addSourceBtn) {
        addSourceBtn.addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pdf,.docx,.pptx,.txt';
            input.onchange = (e) => {
                const file = e.target.files[0];
                if (file) {
                    uploadDocument(file);
                }
            };
            input.click();
        });
    }

    // Chat input
    const sendBtn = document.querySelector('.send-btn');
    const chatInput = document.getElementById('chatInput');
    
    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }
    
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    // Modal close on outside click
    const modal = document.getElementById('actionModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });
    }

    // Update chat input state
    updateChatInputState();
});

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .source-item.disabled {
        opacity: 0.6;
        cursor: not-allowed;
    }
`;
document.head.appendChild(style);
