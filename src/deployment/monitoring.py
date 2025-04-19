"""
Monitoring and logging module.

This module provides functions for monitoring and logging deployed models.
"""

import json
import logging
import os
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from google.cloud import monitoring_v3, logging_v2
import yaml

from src.utils.config import load_config

logger = logging.getLogger(__name__)


class MonitoringClient:
    """Monitoring client for deployed models."""
    
    def __init__(
        self,
        project_id: str,
        location: str = "global",
        credentials: Optional[Any] = None,
    ):
        """
        Initialize monitoring client.

        Args:
            project_id: Google Cloud project ID.
            location: Google Cloud location.
            credentials: Google Cloud credentials.
        """
        self.project_id = project_id
        self.location = location
        self.credentials = credentials
        
        # Initialize monitoring client
        self.client = monitoring_v3.MetricServiceClient(credentials=credentials)
        
        # Initialize project path
        self.project_path = f"projects/{project_id}"
        
        logger.info(f"Initialized monitoring client for project {project_id}")
    
    def create_custom_metric(
        self,
        metric_type: str,
        metric_kind: str,
        value_type: str,
        description: str,
        display_name: Optional[str] = None,
        unit: str = "1",
        labels: Optional[List[Dict[str, str]]] = None,
    ) -> Any:
        """
        Create custom metric.

        Args:
            metric_type: Metric type.
            metric_kind: Metric kind.
            value_type: Value type.
            description: Description.
            display_name: Display name.
            unit: Unit.
            labels: Labels.

        Returns:
            monitoring_v3.MetricDescriptor: Metric descriptor.
        """
        # Set display name if not provided
        if display_name is None:
            display_name = metric_type
        
        # Create metric descriptor
        descriptor = monitoring_v3.types.MetricDescriptor()
        descriptor.type = f"custom.googleapis.com/{metric_type}"
        descriptor.metric_kind = metric_kind
        descriptor.value_type = value_type
        descriptor.description = description
        descriptor.display_name = display_name
        descriptor.unit = unit
        
        # Add labels if provided
        if labels is not None:
            descriptor.labels.extend(labels)
        
        # Create metric descriptor
        descriptor = self.client.create_metric_descriptor(
            name=self.project_path,
            metric_descriptor=descriptor,
        )
        
        logger.info(f"Created custom metric {metric_type}")
        return descriptor
    
    def delete_custom_metric(
        self,
        metric_type: str,
    ) -> None:
        """
        Delete custom metric.

        Args:
            metric_type: Metric type.
        """
        # Delete metric descriptor
        self.client.delete_metric_descriptor(
            name=f"{self.project_path}/metricDescriptors/custom.googleapis.com/{metric_type}",
        )
        
        logger.info(f"Deleted custom metric {metric_type}")
    
    def write_time_series(
        self,
        metric_type: str,
        value: Union[int, float, bool],
        labels: Optional[Dict[str, str]] = None,
        resource_type: str = "global",
        resource_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Write time series.

        Args:
            metric_type: Metric type.
            value: Value.
            labels: Labels.
            resource_type: Resource type.
            resource_labels: Resource labels.
        """
        # Set labels if not provided
        if labels is None:
            labels = {}
        
        # Set resource labels if not provided
        if resource_labels is None:
            resource_labels = {
                "project_id": self.project_id,
            }
        
        # Create time series
        series = monitoring_v3.TimeSeries()
        series.metric.type = f"custom.googleapis.com/{metric_type}"
        
        # Add labels
        for key, value in labels.items():
            series.metric.labels[key] = value
        
        # Add resource
        series.resource.type = resource_type
        
        # Add resource labels
        for key, value in resource_labels.items():
            series.resource.labels[key] = value
        
        # Add point
        point = series.points.add()
        
        # Set value
        if isinstance(value, bool):
            point.value.boolean_value = value
        elif isinstance(value, int):
            point.value.int64_value = value
        else:
            point.value.double_value = value
        
        # Set interval
        now = time.time()
        seconds = int(now)
        nanos = int((now - seconds) * 10**9)
        point.interval.end_time.seconds = seconds
        point.interval.end_time.nanos = nanos
        
        # Write time series
        self.client.create_time_series(
            name=self.project_path,
            time_series=[series],
        )
        
        logger.info(f"Wrote time series for metric {metric_type}")
    
    def read_time_series(
        self,
        metric_type: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        filter_str: Optional[str] = None,
    ) -> List[monitoring_v3.TimeSeries]:
        """
        Read time series.

        Args:
            metric_type: Metric type.
            start_time: Start time.
            end_time: End time.
            filter_str: Filter string.

        Returns:
            List[monitoring_v3.TimeSeries]: Time series.
        """
        # Set start time if not provided
        if start_time is None:
            start_time = datetime.datetime.utcnow() - datetime.timedelta(minutes=30)
        
        # Set end time if not provided
        if end_time is None:
            end_time = datetime.datetime.utcnow()
        
        # Set filter string if not provided
        if filter_str is None:
            filter_str = f'metric.type = "custom.googleapis.com/{metric_type}"'
        
        # Create interval
        interval = monitoring_v3.TimeInterval()
        interval.start_time.seconds = int(start_time.timestamp())
        interval.start_time.nanos = int((start_time.timestamp() - interval.start_time.seconds) * 10**9)
        interval.end_time.seconds = int(end_time.timestamp())
        interval.end_time.nanos = int((end_time.timestamp() - interval.end_time.seconds) * 10**9)
        
        # Read time series
        results = self.client.list_time_series(
            name=self.project_path,
            filter=filter_str,
            interval=interval,
            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        )
        
        # Convert to list
        time_series = list(results)
        
        logger.info(f"Read {len(time_series)} time series for metric {metric_type}")
        return time_series


class LoggingClient:
    """Logging client for deployed models."""
    
    def __init__(
        self,
        project_id: str,
        log_name: str = "model-serving",
        location: str = "global",
        credentials: Optional[Any] = None,
    ):
        """
        Initialize logging client.

        Args:
            project_id: Google Cloud project ID.
            log_name: Log name.
            location: Google Cloud location.
            credentials: Google Cloud credentials.
        """
        self.project_id = project_id
        self.log_name = log_name
        self.location = location
        self.credentials = credentials
        
        # Initialize logging client
        self.client = logging_v2.LoggingServiceV2Client(credentials=credentials)
        
        # Initialize log path
        self.log_path = f"projects/{project_id}/logs/{log_name}"
        
        logger.info(f"Initialized logging client for project {project_id}")
    
    def write_log_entry(
        self,
        payload: Dict[str, Any],
        severity: str = "INFO",
        labels: Optional[Dict[str, str]] = None,
        resource_type: str = "global",
        resource_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Write log entry.

        Args:
            payload: Payload.
            severity: Severity.
            labels: Labels.
            resource_type: Resource type.
            resource_labels: Resource labels.
        """
        # Set labels if not provided
        if labels is None:
            labels = {}
        
        # Set resource labels if not provided
        if resource_labels is None:
            resource_labels = {
                "project_id": self.project_id,
            }
        
        # Create resource
        resource = logging_v2.MonitoredResource(
            type=resource_type,
            labels=resource_labels,
        )
        
        # Create log entry
        entry = logging_v2.LogEntry(
            log_name=self.log_path,
            resource=resource,
            json_payload=payload,
            severity=severity,
            labels=labels,
        )
        
        # Write log entry
        self.client.write_log_entries(
            entries=[entry],
        )
        
        logger.info(f"Wrote log entry to {self.log_path}")
    
    def read_log_entries(
        self,
        filter_str: Optional[str] = None,
        order_by: str = "timestamp desc",
        page_size: int = 100,
    ) -> List[logging_v2.LogEntry]:
        """
        Read log entries.

        Args:
            filter_str: Filter string.
            order_by: Order by.
            page_size: Page size.

        Returns:
            List[logging_v2.LogEntry]: Log entries.
        """
        # Set filter string if not provided
        if filter_str is None:
            filter_str = f'logName = "{self.log_path}"'
        
        # Read log entries
        results = self.client.list_log_entries(
            resource_names=[f"projects/{self.project_id}"],
            filter=filter_str,
            order_by=order_by,
            page_size=page_size,
        )
        
        # Convert to list
        entries = list(results)
        
        logger.info(f"Read {len(entries)} log entries from {self.log_path}")
        return entries


class ModelMonitor:
    """Model monitor for deployed models."""
    
    def __init__(
        self,
        project_id: str,
        model_id: str,
        endpoint_id: str,
        location: str = "global",
        credentials: Optional[Any] = None,
    ):
        """
        Initialize model monitor.

        Args:
            project_id: Google Cloud project ID.
            model_id: Model ID.
            endpoint_id: Endpoint ID.
            location: Google Cloud location.
            credentials: Google Cloud credentials.
        """
        self.project_id = project_id
        self.model_id = model_id
        self.endpoint_id = endpoint_id
        self.location = location
        self.credentials = credentials
        
        # Initialize monitoring client
        self.monitoring_client = MonitoringClient(
            project_id=project_id,
            location=location,
            credentials=credentials,
        )
        
        # Initialize logging client
        self.logging_client = LoggingClient(
            project_id=project_id,
            log_name=f"model-serving-{model_id}",
            location=location,
            credentials=credentials,
        )
        
        logger.info(f"Initialized model monitor for model {model_id}")
    
    def log_prediction(
        self,
        request: Dict[str, Any],
        response: Dict[str, Any],
        latency: float,
        error: Optional[str] = None,
    ) -> None:
        """
        Log prediction.

        Args:
            request: Request.
            response: Response.
            latency: Latency in seconds.
            error: Error message.
        """
        # Create payload
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": self.model_id,
            "endpoint_id": self.endpoint_id,
            "request": request,
            "response": response,
            "latency": latency,
        }
        
        # Add error if provided
        if error is not None:
            payload["error"] = error
            severity = "ERROR"
        else:
            severity = "INFO"
        
        # Write log entry
        self.logging_client.write_log_entry(
            payload=payload,
            severity=severity,
            labels={
                "model_id": self.model_id,
                "endpoint_id": self.endpoint_id,
            },
        )
        
        # Write metrics
        self.monitoring_client.write_time_series(
            metric_type=f"model/{self.model_id}/prediction_count",
            value=1,
            labels={
                "endpoint_id": self.endpoint_id,
                "status": "error" if error is not None else "success",
            },
        )
        
        self.monitoring_client.write_time_series(
            metric_type=f"model/{self.model_id}/prediction_latency",
            value=latency,
            labels={
                "endpoint_id": self.endpoint_id,
            },
        )
    
    def log_feedback(
        self,
        feedback: Dict[str, Any],
    ) -> None:
        """
        Log feedback.

        Args:
            feedback: Feedback.
        """
        # Create payload
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": self.model_id,
            "endpoint_id": self.endpoint_id,
            "feedback": feedback,
        }
        
        # Write log entry
        self.logging_client.write_log_entry(
            payload=payload,
            labels={
                "model_id": self.model_id,
                "endpoint_id": self.endpoint_id,
            },
        )
        
        # Write metrics
        self.monitoring_client.write_time_series(
            metric_type=f"model/{self.model_id}/feedback_count",
            value=1,
            labels={
                "endpoint_id": self.endpoint_id,
            },
        )
        
        # Write feedback score if available
        if "score" in feedback:
            self.monitoring_client.write_time_series(
                metric_type=f"model/{self.model_id}/feedback_score",
                value=feedback["score"],
                labels={
                    "endpoint_id": self.endpoint_id,
                },
            )
    
    def get_prediction_metrics(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get prediction metrics.

        Args:
            start_time: Start time.
            end_time: End time.

        Returns:
            Dict[str, Any]: Prediction metrics.
        """
        # Read prediction count time series
        count_series = self.monitoring_client.read_time_series(
            metric_type=f"model/{self.model_id}/prediction_count",
            start_time=start_time,
            end_time=end_time,
        )
        
        # Read prediction latency time series
        latency_series = self.monitoring_client.read_time_series(
            metric_type=f"model/{self.model_id}/prediction_latency",
            start_time=start_time,
            end_time=end_time,
        )
        
        # Calculate metrics
        metrics = {
            "prediction_count": sum(point.value.int64_value for series in count_series for point in series.points),
            "prediction_latency": {
                "mean": np.mean([point.value.double_value for series in latency_series for point in series.points]) if latency_series else 0,
                "median": np.median([point.value.double_value for series in latency_series for point in series.points]) if latency_series else 0,
                "min": min([point.value.double_value for series in latency_series for point in series.points]) if latency_series else 0,
                "max": max([point.value.double_value for series in latency_series for point in series.points]) if latency_series else 0,
            },
        }
        
        return metrics
    
    def get_feedback_metrics(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get feedback metrics.

        Args:
            start_time: Start time.
            end_time: End time.

        Returns:
            Dict[str, Any]: Feedback metrics.
        """
        # Read feedback count time series
        count_series = self.monitoring_client.read_time_series(
            metric_type=f"model/{self.model_id}/feedback_count",
            start_time=start_time,
            end_time=end_time,
        )
        
        # Read feedback score time series
        score_series = self.monitoring_client.read_time_series(
            metric_type=f"model/{self.model_id}/feedback_score",
            start_time=start_time,
            end_time=end_time,
        )
        
        # Calculate metrics
        metrics = {
            "feedback_count": sum(point.value.int64_value for series in count_series for point in series.points),
            "feedback_score": {
                "mean": np.mean([point.value.double_value for series in score_series for point in series.points]) if score_series else 0,
                "median": np.median([point.value.double_value for series in score_series for point in series.points]) if score_series else 0,
                "min": min([point.value.double_value for series in score_series for point in series.points]) if score_series else 0,
                "max": max([point.value.double_value for series in score_series for point in series.points]) if score_series else 0,
            },
        }
        
        return metrics
    
    def get_prediction_logs(
        self,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get prediction logs.

        Args:
            start_time: Start time.
            end_time: End time.
            limit: Limit.

        Returns:
            List[Dict[str, Any]]: Prediction logs.
        """
        # Create filter string
        filter_parts = [
            f'logName = "{self.logging_client.log_path}"',
            f'labels.model_id = "{self.model_id}"',
            f'labels.endpoint_id = "{self.endpoint_id}"',
        ]
        
        if start_time is not None:
            filter_parts.append(f'timestamp >= "{start_time.isoformat()}"')
        
        if end_time is not None:
            filter_parts.append(f'timestamp <= "{end_time.isoformat()}"')
        
        filter_str = " AND ".join(filter_parts)
        
        # Read log entries
        entries = self.logging_client.read_log_entries(
            filter_str=filter_str,
            page_size=limit,
        )
        
        # Convert to list of dictionaries
        logs = []
        
        for entry in entries:
            log = {
                "timestamp": entry.timestamp.isoformat(),
                "severity": entry.severity,
                "payload": dict(entry.json_payload),
            }
            
            logs.append(log)
        
        return logs


def create_model_monitor_from_config(
    config_path: Union[str, Path],
) -> ModelMonitor:
    """
    Create model monitor from configuration file.

    Args:
        config_path: Path to configuration file.

    Returns:
        ModelMonitor: Model monitor.
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration
    project_id = config.get("gcp", {}).get("project_id")
    location = config.get("gcp", {}).get("location", "global")
    
    model_config = config.get("deployment", {})
    model_id = model_config.get("model_id")
    endpoint_id = model_config.get("endpoint_id")
    
    # Create model monitor
    monitor = ModelMonitor(
        project_id=project_id,
        model_id=model_id,
        endpoint_id=endpoint_id,
        location=location,
    )
    
    return monitor


def setup_monitoring(
    config_path: Union[str, Path],
) -> None:
    """
    Set up monitoring.

    Args:
        config_path: Path to configuration file.
    """
    # Create model monitor
    monitor = create_model_monitor_from_config(config_path)
    
    # Create custom metrics
    monitor.monitoring_client.create_custom_metric(
        metric_type=f"model/{monitor.model_id}/prediction_count",
        metric_kind="CUMULATIVE",
        value_type="INT64",
        description=f"Number of predictions for model {monitor.model_id}",
        display_name=f"Prediction Count - {monitor.model_id}",
        labels=[
            {
                "key": "endpoint_id",
                "value_type": "STRING",
                "description": "Endpoint ID",
            },
            {
                "key": "status",
                "value_type": "STRING",
                "description": "Prediction status (success or error)",
            },
        ],
    )
    
    monitor.monitoring_client.create_custom_metric(
        metric_type=f"model/{monitor.model_id}/prediction_latency",
        metric_kind="GAUGE",
        value_type="DOUBLE",
        description=f"Prediction latency for model {monitor.model_id}",
        display_name=f"Prediction Latency - {monitor.model_id}",
        unit="s",
        labels=[
            {
                "key": "endpoint_id",
                "value_type": "STRING",
                "description": "Endpoint ID",
            },
        ],
    )
    
    monitor.monitoring_client.create_custom_metric(
        metric_type=f"model/{monitor.model_id}/feedback_count",
        metric_kind="CUMULATIVE",
        value_type="INT64",
        description=f"Number of feedback submissions for model {monitor.model_id}",
        display_name=f"Feedback Count - {monitor.model_id}",
        labels=[
            {
                "key": "endpoint_id",
                "value_type": "STRING",
                "description": "Endpoint ID",
            },
        ],
    )
    
    monitor.monitoring_client.create_custom_metric(
        metric_type=f"model/{monitor.model_id}/feedback_score",
        metric_kind="GAUGE",
        value_type="DOUBLE",
        description=f"Feedback score for model {monitor.model_id}",
        display_name=f"Feedback Score - {monitor.model_id}",
        labels=[
            {
                "key": "endpoint_id",
                "value_type": "STRING",
                "description": "Endpoint ID",
            },
        ],
    )
    
    logger.info(f"Set up monitoring for model {monitor.model_id}")


def teardown_monitoring(
    config_path: Union[str, Path],
) -> None:
    """
    Tear down monitoring.

    Args:
        config_path: Path to configuration file.
    """
    # Create model monitor
    monitor = create_model_monitor_from_config(config_path)
    
    # Delete custom metrics
    monitor.monitoring_client.delete_custom_metric(
        metric_type=f"model/{monitor.model_id}/prediction_count",
    )
    
    monitor.monitoring_client.delete_custom_metric(
        metric_type=f"model/{monitor.model_id}/prediction_latency",
    )
    
    monitor.monitoring_client.delete_custom_metric(
        metric_type=f"model/{monitor.model_id}/feedback_count",
    )
    
    monitor.monitoring_client.delete_custom_metric(
        metric_type=f"model/{monitor.model_id}/feedback_score",
    )
    
    logger.info(f"Tore down monitoring for model {monitor.model_id}")
