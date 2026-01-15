// Toggle Left/Right Panels
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
    
    // Manage grid classes
    workspace.classList.remove('left-collapsed', 'right-collapsed', 'both-collapsed');
    
    if (isLeftCollapsed && isRightCollapsed) {
        workspace.classList.add('both-collapsed');
    } else if (isLeftCollapsed) {
        workspace.classList.add('left-collapsed');
    } else if (isRightCollapsed) {
        workspace.classList.add('right-collapsed');
    }

    // Toggle Buttons visibility
    leftBtn.style.display = isLeftCollapsed ? 'flex' : 'none';
    rightBtn.style.display = isRightCollapsed ? 'flex' : 'none';
}

// Source Selection (Multiple)
function toggleSource(element) {
    element.classList.toggle('active');
    updateSourceCount();
}

function updateSourceCount() {
    const activeSources = document.querySelectorAll('.source-item.active').length;
    document.getElementById('sourceCount').textContent = activeSources;
}

// Config
const modalTitles = {
    'summary': 'Generate Document Summary',
    'qna': 'Create Practice Q&A',
    'tips': 'Extract Interview Tips',
    'slides': 'Customize Slide Deck'
};

const defaultInstructions = {
    'summary': 'Focus on the key findings and financial metrics.',
    'qna': 'Create a mix of technical and behavioral questions.',
    'tips': 'Highlight areas where I can demonstrate leadership.',
    'slides': 'Create a 10-slide deck for a managerial audience.'
};

let currentAction = '';

// Modal Functions
function openModal(actionType) {
    currentAction = actionType;
    const modal = document.getElementById('actionModal');
    const title = document.getElementById('modalTitle');
    const instructions = document.getElementById('modalInstructions');

    // Update Content
    title.textContent = modalTitles[actionType];
    instructions.placeholder = defaultInstructions[actionType];
    instructions.value = ''; // Clear previous input

    // Show Modal
    modal.classList.add('active');
}

function closeModal() {
    const modal = document.getElementById('actionModal');
    modal.classList.remove('active');
}

// Generate Action
function startGeneration() {
    closeModal();
    
    // Create new history item
    const historyList = document.getElementById('historyList');
    const item = document.createElement('div');
    item.className = 'history-item generating';
    
    // Determine icon based on action
    let iconClass = 'fa-file-lines';
    if(currentAction === 'qna') iconClass = 'fa-clipboard-question';
    if(currentAction === 'tips') iconClass = 'fa-lightbulb';
    if(currentAction === 'slides') iconClass = 'fa-presentation-screen';

    item.innerHTML = `
        <div class="history-icon"><i class="fa-solid ${iconClass}"></i></div>
        <div class="history-content">
            <span class="history-title">${modalTitles[currentAction]}</span>
            <span class="history-date">Generating...</span>
        </div>
        <div class="history-loader"><i class="fa-solid fa-spinner fa-spin"></i></div>
    `;

    // Add to top of list
    historyList.insertBefore(item, historyList.firstChild);

    // Simulate API delay
    setTimeout(() => {
        item.classList.remove('generating');
        item.querySelector('.history-date').textContent = 'Just now';
        item.querySelector('.history-loader').innerHTML = '<i class="fa-solid fa-chevron-right"></i>';
        item.classList.add('completed');
    }, 2500);
}

// Close modal on outside click
document.getElementById('actionModal').addEventListener('click', (e) => {
    if (e.target === document.getElementById('actionModal')) {
        closeModal();
    }
});

// Add Source (Mock)
document.querySelector('.add-source-btn').addEventListener('click', () => {
    const list = document.querySelector('.sources-list');
    const newSource = document.createElement('div');
    newSource.className = 'source-item active'; // Auto-select new
    newSource.onclick = function() { toggleSource(this); }; // Add click handler
    
    newSource.innerHTML = `
        <i class="fa-regular fa-file-word source-icon" style="color: #2b579a;"></i>
        <div class="source-info">
            <span class="source-name">New_Document.docx</span>
            <span class="source-meta">Uploading...</span>
        </div>
        <div class="selection-indicator"><i class="fa-solid fa-spinner fa-spin"></i></div>
    `;
    list.appendChild(newSource);
    updateSourceCount();

    setTimeout(() => {
        newSource.querySelector('.source-meta').textContent = 'Parsed • 450 KB';
        const indicator = newSource.querySelector('.selection-indicator');
        indicator.innerHTML = '<i class="fa-solid fa-check"></i>';
    }, 1500);
});

// Chat (Simple Echo Mock)
document.querySelector('.send-btn').addEventListener('click', sendMessage);
document.getElementById('chatInput').addEventListener('keypress', (e) => {
    if(e.key === 'Enter') sendMessage();
});

function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if(!text) return;

    const chatArea = document.getElementById('chatArea');
    
    // Remove empty state if present
    const emptyState = document.querySelector('.empty-state');
    if(emptyState) emptyState.style.display = 'none';

    // User Message
    const userMsg = document.createElement('div');
    userMsg.style.cssText = 'background: #eff6ff; padding: 12px 16px; border-radius: 12px 12px 0 12px; align-self: flex-end; max-width: 80%; margin-bottom: 12px; font-size: 14px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);';
    userMsg.textContent = text;
    chatArea.appendChild(userMsg);

    input.value = '';

    // Create Loading Bubble
    const loadingMsg = document.createElement('div');
    loadingMsg.className = 'chat-message ai-loading';
    loadingMsg.style.cssText = 'background: white; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 12px 12px 12px 0; align-self: flex-start; max-width: 80%; margin-bottom: 12px; font-size: 14px; display: flex; align-items: center; gap: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);';
    loadingMsg.innerHTML = `
        <i class="fa-solid fa-wand-magic-sparkles" style="color: #6366f1;"></i>
        <span style="color: #6b7280;">Thinking...</span>
    `;
    chatArea.appendChild(loadingMsg);
    chatArea.scrollTop = chatArea.scrollHeight;

    // Simulate Processing Delay (3 seconds)
    setTimeout(() => {
        // Remove loading bubble
        loadingMsg.remove();

        const aiMsg = document.createElement('div');
        aiMsg.style.cssText = 'background: white; border: 1px solid #e0e0e0; padding: 12px 16px; border-radius: 12px 12px 12px 0; align-self: flex-start; max-width: 80%; margin-bottom: 12px; font-size: 14px; line-height: 1.5; box-shadow: 0 1px 2px rgba(0,0,0,0.05); animation: fadeIn 0.3s ease;';
        
        // Simulated smart response
        aiMsg.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-weight: 500; color: #6366f1;">
                <i class="fa-solid fa-wand-magic-sparkles"></i> InsightForge
            </div>
            <p>Based on the selected documents, here's what I found regarding "<strong>${text}</strong>":</p>
            <ul style="margin: 8px 0 8px 20px; padding: 0;">
                <li>The corporate finance assignment highlights key valuation metrics including NPV and IRR.</li>
                <li>Market sizing analysis suggests a Total Addressable Market (TAM) of $4.5B by 2025.</li>
            </ul>
            <p>Would you like me to elaborate on the specific financial models used?</p>
            <div style="margin-top: 8px; font-size: 11px; color: #6b7280; display: flex; gap: 12px;">
                <span><i class="fa-regular fa-file-pdf"></i> Corporate_Finance...</span>
                <span><i class="fa-regular fa-file-powerpoint"></i> Market_Sizing...</span>
            </div>
        `;
        
        chatArea.appendChild(aiMsg);
        chatArea.scrollTop = chatArea.scrollHeight;
    }, 3000); // 3 seconds delay
}

