/**
 * 進捗表示用JavaScript
 */

class ProgressTracker {
    constructor(sessionId) {
        this.sessionId = sessionId;
        this.pollInterval = 1000; // 1秒ごとにポーリング
        this.pollTimer = null;
        this.isPolling = false;
    }
    
    start() {
        // モデル学習を開始
        this.startTraining();
        
        // ポーリングを開始
        this.isPolling = true;
        this.poll();
    }
    
    async startTraining() {
        try {
            const response = await fetch('/train-models-start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            
            const data = await response.json();
            if (data.error) {
                this.showError(data.error);
            }
        } catch (error) {
            console.error('Training start error:', error);
            this.showError('モデル学習の開始に失敗しました');
        }
    }
    
    async poll() {
        if (!this.isPolling) {
            return;
        }
        
        try {
            const response = await fetch(`/api/progress?session_id=${this.sessionId}`);
            const data = await response.json();
            
            if (data.error) {
                this.showError(data.error);
                this.stop();
                return;
            }
            
            this.updateUI(data);
            
            // 完了またはエラーの場合、ポーリングを停止
            if (data.status === 'completed') {
                this.stop();
                this.showCompletion();
            } else if (data.status === 'error') {
                this.stop();
                this.showError(data.error_message || 'エラーが発生しました');
            } else {
                // 次のポーリングをスケジュール
                this.pollTimer = setTimeout(() => this.poll(), this.pollInterval);
            }
        } catch (error) {
            console.error('Polling error:', error);
            // エラーが発生してもポーリングを続ける
            this.pollTimer = setTimeout(() => this.poll(), this.pollInterval);
        }
    }
    
    stop() {
        this.isPolling = false;
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
    }
    
    updateUI(data) {
        // 全体の進捗バーを更新
        const progressPercentage = data.progress_percentage || 0;
        const progressBar = document.getElementById('overall-progress');
        const progressText = document.getElementById('progress-text');
        
        if (progressBar) {
            progressBar.style.width = `${progressPercentage}%`;
            progressBar.setAttribute('aria-valuenow', progressPercentage);
        }
        
        if (progressText) {
            progressText.textContent = `${Math.round(progressPercentage)}%`;
        }
        
        // 経過時間を更新
        const elapsedTime = document.getElementById('elapsed-time');
        if (elapsedTime) {
            elapsedTime.textContent = this.formatTime(data.elapsed_time || 0);
        }
        
        // 残り時間を更新
        const remainingTime = document.getElementById('remaining-time');
        if (remainingTime) {
            const estimated = data.estimated_remaining_time;
            if (estimated !== null && estimated !== undefined) {
                remainingTime.textContent = this.formatTime(estimated);
            } else {
                remainingTime.textContent = '計算中...';
            }
        }
        
        // ステップを更新
        this.updateSteps(data.steps || [], data.current_step);
    }
    
    updateSteps(steps, currentStepName) {
        const container = document.getElementById('steps-container');
        if (!container) {
            return;
        }
        
        // 既存のステップをクリア
        container.innerHTML = '';
        
        steps.forEach((step, index) => {
            const stepDiv = document.createElement('div');
            stepDiv.className = `progress-step ${step.status}`;
            
            const icon = this.getStepIcon(step.status);
            const name = step.name || `ステップ ${index + 1}`;
            const duration = step.duration ? ` (${this.formatTime(step.duration)})` : '';
            
            stepDiv.innerHTML = `
                <div>
                    <span class="step-icon">${icon}</span>
                    <strong>${name}</strong>${duration}
                </div>
            `;
            
            container.appendChild(stepDiv);
        });
    }
    
    getStepIcon(status) {
        switch (status) {
            case 'completed':
                return '✓';
            case 'running':
                return '<span class="spinner-border spinner-border-sm" role="status"></span>';
            case 'error':
                return '✗';
            default:
                return '○';
        }
    }
    
    formatTime(seconds) {
        if (seconds < 60) {
            return `${Math.round(seconds)}秒`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${minutes}分${secs}秒`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}時間${minutes}分`;
        }
    }
    
    showError(message) {
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
    }
    
    showCompletion() {
        const completionDiv = document.getElementById('completion-message');
        if (completionDiv) {
            completionDiv.style.display = 'block';
        }
        
        // プログレスバーのアニメーションを停止
        const progressBar = document.getElementById('overall-progress');
        if (progressBar) {
            progressBar.classList.remove('progress-bar-animated');
        }
    }
}

