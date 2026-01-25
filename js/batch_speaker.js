import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-TD-Qwen3TTS.BatchGenerateSpeaker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TDQwen3TTSBatchGenerateSpeaker") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Find the speakers_config widget
                const configWidget = this.widgets.find(w => w.name === "speakers_config");

                // Add "Manage Speakers" button
                this.addWidget("button", "Manage Speakers", "Edit", (w) => {
                    showSpeakerDialog(this, configWidget);
                });

                return r;
            };

            // Handle execution results to get generated audio
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                
                // message is the output object. message.batch_outputs comes from our python "ui" return
                if (message && message.batch_outputs) {
                    this.batchOutputs = message.batch_outputs;
                    // If the dialog is open, we could potentially refresh it, 
                    // but for now let's just let the user re-open or if we implement auto-refresh later.
                    // A simple check:
                    const dialog = document.getElementById('td-qwen3-speaker-dialog');
                    if (dialog && dialog.dataset.nodeId == this.id) {
                         // We can trigger a refresh if we exposed the render function
                         // For now, we'll just leave it.
                    }
                }
            };
        }
    }
});

function showSpeakerDialog(node, configWidget) {
    // Remove existing if any
    const existing = document.getElementById('td-qwen3-speaker-dialog');
    if (existing) document.body.removeChild(existing);

    const dialog = document.createElement("div");
    dialog.id = 'td-qwen3-speaker-dialog';
    dialog.dataset.nodeId = node.id;
    dialog.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: #222;
        color: #ddd;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 0 20px rgba(0,0,0,0.5);
        z-index: 10000;
        width: 900px;
        max-height: 85vh;
        display: flex;
        flex-direction: column;
        border: 1px solid #444;
        font-family: sans-serif;
    `;

    const title = document.createElement("h3");
    title.innerText = "Manage Speakers (Batch Generate)";
    title.style.marginTop = "0";
    dialog.appendChild(title);

    const contentArea = document.createElement("div");
    contentArea.style.cssText = `
        flex: 1;
        overflow-y: auto;
        margin-bottom: 20px;
        border: 1px solid #333;
        padding: 10px;
        background: #111;
    `;
    dialog.appendChild(contentArea);

    // Load existing config
    let speakers = [];
    try {
        if (configWidget && configWidget.value) {
            speakers = JSON.parse(configWidget.value);
        }
    } catch (e) {
        console.error("Invalid JSON in speakers_config", e);
    }
    if (!Array.isArray(speakers)) speakers = [];

    // Function to render list
    function renderList() {
        contentArea.innerHTML = "";
        if (speakers.length === 0) {
            contentArea.innerHTML = "<div style='padding:20px; text-align:center; color:#666;'>No speakers defined. Click 'Add Speaker' to start.</div>";
            return;
        }

        speakers.forEach((spk, index) => {
            const row = document.createElement("div");
            row.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 10px;
                margin-bottom: 10px;
                background: #2a2a2a;
                padding: 10px;
                border-radius: 4px;
            `;

            const topRow = document.createElement("div");
            topRow.style.cssText = "display: flex; gap: 10px; align-items: flex-start; width: 100%;";
            row.appendChild(topRow);

            // Name
            const nameGroup = document.createElement("div");
            nameGroup.style.flex = "1";
            nameGroup.innerHTML = "<label style='display:block;font-size:12px;color:#aaa;margin-bottom:4px'>Name</label>";
            const nameInput = document.createElement("input");
            nameInput.value = spk.name || "";
            nameInput.style.cssText = "width:100%; background:#333; color:#fff; border:1px solid #555; padding:4px;";
            nameInput.onchange = (e) => { spk.name = e.target.value; };
            nameGroup.appendChild(nameInput);
            topRow.appendChild(nameGroup);

            // Instruct
            const instructGroup = document.createElement("div");
            instructGroup.style.flex = "2";
            instructGroup.innerHTML = "<label style='display:block;font-size:12px;color:#aaa;margin-bottom:4px'>Voice Style (Instruct)</label>";
            const instructInput = document.createElement("textarea");
            instructInput.value = spk.instruct || "";
            instructInput.rows = 2;
            instructInput.style.cssText = "width:100%; background:#333; color:#fff; border:1px solid #555; padding:4px; resize:vertical;";
            instructInput.onchange = (e) => { spk.instruct = e.target.value; };
            instructGroup.appendChild(instructInput);
            topRow.appendChild(instructGroup);

            // Text
            const textGroup = document.createElement("div");
            textGroup.style.flex = "2";
            textGroup.innerHTML = "<label style='display:block;font-size:12px;color:#aaa;margin-bottom:4px'>Sample Text</label>";
            const textInput = document.createElement("textarea");
            textInput.value = spk.text || "";
            textInput.rows = 2;
            textInput.style.cssText = "width:100%; background:#333; color:#fff; border:1px solid #555; padding:4px; resize:vertical;";
            textInput.onchange = (e) => { spk.text = e.target.value; };
            textGroup.appendChild(textInput);
            topRow.appendChild(textGroup);

            // Delete
            const delBtn = document.createElement("button");
            delBtn.innerText = "X";
            delBtn.style.cssText = "background:#800; color:#fff; border:none; padding:5px 10px; cursor:pointer; margin-top:20px;";
            delBtn.onclick = () => {
                speakers.splice(index, 1);
                renderList();
            };
            topRow.appendChild(delBtn);

            // Preview Area (if audio exists)
            // We look up audio from node.batchOutputs by name
            // Note: If names are duplicated, this might pick the first match.
            if (node.batchOutputs && spk.name) {
                // Find latest output for this name? 
                // Or just find one. 
                // Since batchOutputs might accumulate if we don't clear it? 
                // Actually batchOutputs is replaced on each execution.
                const audioInfo = node.batchOutputs.find(o => o.name === spk.name);
                
                if (audioInfo) {
                    const audioRow = document.createElement("div");
                    audioRow.style.cssText = "margin-top: 5px; background: #222; padding: 5px; border-radius: 4px; display: flex; align-items: center; gap: 10px;";
                    
                    const label = document.createElement("span");
                    label.innerText = "Last Generated Preview:";
                    label.style.fontSize = "12px";
                    label.style.color = "#aaa";
                    audioRow.appendChild(label);

                    const audioPlayer = document.createElement("audio");
                    audioPlayer.controls = true;
                    audioPlayer.style.height = "30px";
                    // Construct URL: /view?filename=...&type=...&subfolder=...
                    const params = new URLSearchParams({
                        filename: audioInfo.filename,
                        type: audioInfo.type,
                        subfolder: audioInfo.subfolder || ""
                    });
                    audioPlayer.src = "/view?" + params.toString();
                    audioRow.appendChild(audioPlayer);
                    
                    row.appendChild(audioRow);
                }
            }

            contentArea.appendChild(row);
        });
    }

    renderList();

    // Buttons area
    const buttons = document.createElement("div");
    buttons.style.cssText = "display:flex; justify-content:space-between; align-items:center;";
    dialog.appendChild(buttons);

    const leftBtns = document.createElement("div");
    buttons.appendChild(leftBtns);

    const addBtn = document.createElement("button");
    addBtn.innerText = "+ Add Speaker";
    addBtn.style.cssText = "background:#444; color:#fff; border:none; padding:8px 16px; cursor:pointer; margin-right:10px;";
    addBtn.onclick = () => {
        speakers.push({ name: "NewSpeaker", instruct: "A clear voice", text: "Hello world" });
        renderList();
        // Scroll to bottom
        setTimeout(() => contentArea.scrollTop = contentArea.scrollHeight, 10);
    };
    leftBtns.appendChild(addBtn);

    const rightBtns = document.createElement("div");
    buttons.appendChild(rightBtns);

    const cancelBtn = document.createElement("button");
    cancelBtn.innerText = "Cancel";
    cancelBtn.style.cssText = "background:transparent; color:#aaa; border:1px solid #444; padding:8px 16px; cursor:pointer; margin-right:10px;";
    cancelBtn.onclick = () => {
        document.body.removeChild(dialog);
    };
    rightBtns.appendChild(cancelBtn);

    const saveBtn = document.createElement("button");
    saveBtn.innerText = "Save";
    saveBtn.style.cssText = "background:#28a745; color:#fff; border:none; padding:8px 24px; cursor:pointer; font-weight:bold;";
    saveBtn.onclick = () => {
        if (configWidget) {
            configWidget.value = JSON.stringify(speakers, null, 2);
            // Trigger graph change
            if (node.onResize) node.onResize(node.size);
            app.graph.setDirtyCanvas(true, true);
        }
        document.body.removeChild(dialog);
    };
    rightBtns.appendChild(saveBtn);

    document.body.appendChild(dialog);
}
