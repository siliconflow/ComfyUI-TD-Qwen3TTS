import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "ComfyUI-TD-Qwen3TTS.MultiDialog",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "TDQwen3TTSMultiDialog") {
			const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, slot) {
				const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined;
                
                // Ensure inputs exist
                // 确保输入存在
                if (!this.inputs) return r;

                // 1. Identify all speaker inputs and their indices
                // 1. 识别所有 speaker 输入及其索引
                const speakerInputs = [];
                for(let i=0; i<this.inputs.length; i++) {
                    if(this.inputs[i].name.startsWith("speaker_")) {
                        speakerInputs.push({
                            index: i,
                            name: this.inputs[i].name,
                            input: this.inputs[i],
                            num: parseInt(this.inputs[i].name.split("_")[1])
                        });
                    }
                }

                // 2. Find the highest numbered speaker slot that is currently connected
                // 2. 找到当前已连接的最大编号 speaker 插槽
                let maxConnectedNum = 0;
                let maxExistingNum = 0;

                for(const item of speakerInputs) {
                    if(item.num > maxExistingNum) maxExistingNum = item.num;
                    if(item.input.link !== undefined && item.input.link !== null) {
                        if(item.num > maxConnectedNum) maxConnectedNum = item.num;
                        
                        // NEW LOGIC: Rename input based on upstream node name
                        // 新逻辑：根据上游节点名称重命名输入
                        const linkId = item.input.link;
                        const link = app.graph.links[linkId];
                        if (link) {
                            const originNode = app.graph.getNodeById(link.origin_id);
                            if (originNode && originNode.widgets) {
                                // Look for 'name' widget
                                // 查找 'name' 控件
                                const nameWidget = originNode.widgets.find(w => w.name === "name");
                                if (nameWidget && nameWidget.value) {
                                    const newLabel = `${nameWidget.value}`;
                                    // Only update if changed to avoid loop/redraw issues
                                    // 仅在发生变化时更新，以避免循环/重绘问题
                                    if (item.input.label !== newLabel) {
                                        item.input.label = newLabel;
                                        // Force redraw of this node
                                        // 强制重绘此节点
                                        if (this.setDirtyCanvas) {
                                            this.setDirtyCanvas(true, true);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Reset label if disconnected
                        // 如果断开连接，重置标签
                         if (item.input.label) {
                            item.input.label = undefined; // Or null, to revert to name // 或者设为 null，以恢复为名称
                            if (this.setDirtyCanvas) {
                                this.setDirtyCanvas(true, true);
                            }
                        }
                    }
                }

                // 3. Determine target behavior
                // 3. 确定目标行为
                // We want to ensure that if speaker_N is connected, speaker_{N+1} exists.
                // 我们要确保如果 speaker_N 已连接，则 speaker_{N+1} 存在。
                // If nothing is connected, we want speaker_1 to exist.
                // 如果没有任何连接，我们要确保 speaker_1 存在。
                const targetNum = maxConnectedNum + 1;

                // 4. Add if missing
                // 4. 如果缺失则添加
                if (maxExistingNum < targetNum) {
                    // Add intermediate slots if for some reason we jumped (unlikely but safe)
                    // 如果由于某种原因跳过了（不太可能但为了安全起见），添加中间插槽
                    for (let i = maxExistingNum + 1; i <= targetNum; i++) {
                        // console.log("Adding input", `speaker_${i}`);
                        this.addInput(`speaker_${i}`, "SPEAKER");
                    }
                }
                
                // 5. Remove if too many (trailing unconnected slots)
                // 5. 如果过多则移除（末尾未连接的插槽）
                // We only want to keep up to targetNum.
                // 我们只想保留到 targetNum。
                // Any slot > targetNum should be removed IF it is unconnected.
                // 任何 > targetNum 的插槽如果未连接都应被移除。
                // We iterate backwards to safely remove by index.
                // 我们反向迭代以按索引安全移除。
                
                // Refresh inputs list as we might have added some
                // 刷新输入列表，因为我们可能添加了一些
                // Actually, let's just use the current logic: 
                // 实际上，我们就使用当前的逻辑：
                // Any input named speaker_X where X > targetNum is a candidate for removal.
                // 任何名为 speaker_X 且 X > targetNum 的输入都是移除的候选对象。
                
                // Note: Removing inputs changes indices, so we must be careful.
                // 注意：移除输入会改变索引，所以我们必须小心。
                // It's best to grab the current list of inputs again.
                // 最好再次获取当前的输入列表。
                
                let inputsModified = false;
                
                // Loop backwards
                // 反向循环
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    const inp = this.inputs[i];
                    if (inp.name.startsWith("speaker_")) {
                        const num = parseInt(inp.name.split("_")[1]);
                        if (num > targetNum) {
                            // Verify it's not connected (double check)
                            // 验证它未连接（再次检查）
                            if (inp.link === undefined || inp.link === null) {
                                // console.log("Removing input", inp.name);
                                this.removeInput(i);
                                inputsModified = true;
                            }
                        }
                    }
                }
                
                // If we modified inputs, we might need to notify graph update? 
                // 如果我们修改了输入，可能需要通知图形更新？
                // Usually not strictly required for inputs but good practice if visual glitches occur.
                // 通常对于输入不是严格要求的，但如果出现视觉故障，这是一个好的做法。
                
				return r;
			};
		}
	}
});
