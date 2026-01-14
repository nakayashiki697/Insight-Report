"""
進捗管理モジュール
プログレスバー表示用の進捗情報を管理
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime


class ProgressTracker:
    """進捗状況を追跡するクラス"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.steps = []
        self.current_step_index = 0
        self.start_time = None
        self.current_step_start_time = None
        self.status = 'pending'  # pending, running, completed, error
        self.error_message = None
        self.estimated_total_time = None
        
    def start(self):
        """処理開始"""
        self.start_time = time.time()
        self.status = 'running'
    
    def add_step(self, name: str, estimated_percentage: int = 0):
        """
        ステップを追加
        
        Args:
            name: ステップ名
            estimated_percentage: 推定進捗率（0-100）
        """
        self.steps.append({
            'name': name,
            'status': 'pending',  # pending, running, completed, error
            'estimated_percentage': estimated_percentage,
            'start_time': None,
            'end_time': None,
            'duration': None
        })
    
    def start_step(self, step_index: int):
        """ステップを開始"""
        if 0 <= step_index < len(self.steps):
            self.current_step_index = step_index
            self.steps[step_index]['status'] = 'running'
            self.steps[step_index]['start_time'] = time.time()
            self.current_step_start_time = time.time()
    
    def complete_step(self, step_index: int):
        """ステップを完了"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.steps[step_index]['end_time'] = time.time()
            if self.steps[step_index]['start_time']:
                self.steps[step_index]['duration'] = (
                    self.steps[step_index]['end_time'] - 
                    self.steps[step_index]['start_time']
                )
    
    def set_error(self, error_message: str):
        """エラーを設定"""
        self.status = 'error'
        self.error_message = error_message
        if self.current_step_index < len(self.steps):
            self.steps[self.current_step_index]['status'] = 'error'
    
    def complete(self):
        """処理完了"""
        self.status = 'completed'
        # 未完了のステップを完了にする
        for i, step in enumerate(self.steps):
            if step['status'] == 'running':
                self.complete_step(i)
    
    def get_progress_percentage(self) -> float:
        """現在の進捗率を取得（0-100）"""
        if not self.steps:
            return 0.0
        
        total_percentage = 0.0
        for i, step in enumerate(self.steps):
            if step['status'] == 'completed':
                total_percentage += step['estimated_percentage']
            elif step['status'] == 'running':
                # 実行中のステップは50%完了とみなす
                total_percentage += step['estimated_percentage'] * 0.5
        
        return min(total_percentage, 100.0)
    
    def get_elapsed_time(self) -> float:
        """経過時間を取得（秒）"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def estimate_remaining_time(self) -> Optional[float]:
        """残り時間を推定（秒）"""
        if self.status != 'running' or not self.steps:
            return None
        
        elapsed = self.get_elapsed_time()
        if elapsed == 0:
            return None
        
        # 完了したステップの実際の処理時間を集計
        completed_time = 0.0
        completed_percentage = 0.0
        
        for step in self.steps:
            if step['status'] == 'completed' and step.get('duration'):
                completed_time += step['duration']
                completed_percentage += step['estimated_percentage']
            elif step['status'] == 'running' and step.get('start_time'):
                # 実行中のステップは経過時間を追加
                running_duration = time.time() - step['start_time']
                completed_time += running_duration
                # 実行中のステップは50%完了とみなす
                completed_percentage += step['estimated_percentage'] * 0.5
        
        if completed_percentage == 0:
            return None
        
        # 完了したステップの処理速度（% per second）を計算
        if completed_time > 0:
            speed = completed_percentage / completed_time  # % per second
        else:
            return None
        
        # 残りの進捗率を計算
        remaining_percentage = 100.0 - completed_percentage
        
        if remaining_percentage <= 0:
            return 0.0
        
        # 残り時間を推定
        if speed > 0:
            estimated_remaining = remaining_percentage / speed
        else:
            # 速度が計算できない場合、完了したステップの平均時間から推定
            if completed_percentage > 0:
                avg_time_per_percent = completed_time / completed_percentage
                estimated_remaining = remaining_percentage * avg_time_per_percent
            else:
                return None
        
        return max(0.0, estimated_remaining)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（JSONシリアライズ可能）"""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'progress_percentage': self.get_progress_percentage(),
            'elapsed_time': self.get_elapsed_time(),
            'estimated_remaining_time': self.estimate_remaining_time(),
            'current_step': self.steps[self.current_step_index]['name'] if self.steps and self.current_step_index < len(self.steps) else None,
            'steps': [
                {
                    'name': step['name'],
                    'status': step['status'],
                    'estimated_percentage': step['estimated_percentage'],
                    'duration': step.get('duration')
                }
                for step in self.steps
            ],
            'error_message': self.error_message
        }


# グローバルな進捗トラッカーのストレージ（本番環境ではRedis等を使用）
_progress_trackers: Dict[str, ProgressTracker] = {}


def get_progress_tracker(session_id: str) -> Optional[ProgressTracker]:
    """進捗トラッカーを取得"""
    return _progress_trackers.get(session_id)


def create_progress_tracker(session_id: str) -> ProgressTracker:
    """進捗トラッカーを作成"""
    tracker = ProgressTracker(session_id)
    _progress_trackers[session_id] = tracker
    return tracker


def remove_progress_tracker(session_id: str):
    """進捗トラッカーを削除"""
    if session_id in _progress_trackers:
        del _progress_trackers[session_id]

