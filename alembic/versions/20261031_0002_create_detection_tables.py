"""Create durable detection runs and anomaly results.

Revision ID: 20261031_0002
Revises: 20261030_0001
Create Date: 2026-10-31
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "20261031_0002"
down_revision: str | None = "20261030_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "detection_runs",
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("meter_id", sa.Integer(), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("start_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("lease_token", sa.Uuid(as_uuid=True), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("training_readings", sa.Integer(), server_default="0", nullable=False),
        sa.Column("readings_processed", sa.Integer(), server_default="0", nullable=False),
        sa.Column("anomaly_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("normal_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("unavailable_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("configuration", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.String(length=512), nullable=True),
        sa.CheckConstraint(
            "algorithm IN ('global_iqr', 'rolling_mean_std', 'time_of_day', 'isolation_forest')",
            name="ck_detection_runs_valid_algorithm",
        ),
        sa.CheckConstraint(
            "readings_processed = anomaly_count + normal_count + unavailable_count",
            name="ck_detection_runs_prediction_counts_reconcile",
        ),
        sa.CheckConstraint(
            "start_timestamp IS NULL OR end_timestamp IS NULL "
            "OR start_timestamp < end_timestamp",
            name="ck_detection_runs_valid_requested_range",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_detection_runs_valid_status",
        ),
        sa.ForeignKeyConstraint(
            ["meter_id"],
            ["meters.id"],
            name="fk_detection_runs_meter_id_meters",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_detection_runs"),
    )
    op.create_index(
        "ix_detection_runs_status_lease_expires_at",
        "detection_runs",
        ["status", "lease_expires_at"],
    )
    op.create_index(
        "ix_detection_runs_meter_id_created_at",
        "detection_runs",
        ["meter_id", "created_at"],
    )
    op.create_index(
        "ix_detection_runs_algorithm_created_at",
        "detection_runs",
        ["algorithm", "created_at"],
    )

    op.create_table(
        "anomaly_results",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("detection_run_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("reading_id", sa.Integer(), nullable=False),
        sa.Column("prediction_status", sa.String(length=16), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("anomaly_category", sa.String(length=100), nullable=True),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "prediction_status IN ('normal', 'anomaly', 'unavailable')",
            name="ck_anomaly_results_valid_prediction_status",
        ),
        sa.ForeignKeyConstraint(
            ["detection_run_id"],
            ["detection_runs.id"],
            name="fk_anomaly_results_detection_run_id_detection_runs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["reading_id"],
            ["meter_readings.id"],
            name="fk_anomaly_results_reading_id_meter_readings",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_anomaly_results"),
        sa.UniqueConstraint(
            "detection_run_id",
            "reading_id",
            name="uq_anomaly_results_run_reading",
        ),
    )
    op.create_index(
        "ix_anomaly_results_run_status",
        "anomaly_results",
        ["detection_run_id", "prediction_status"],
    )
    op.create_index("ix_anomaly_results_reading_id", "anomaly_results", ["reading_id"])


def downgrade() -> None:
    op.drop_index("ix_anomaly_results_reading_id", table_name="anomaly_results")
    op.drop_index("ix_anomaly_results_run_status", table_name="anomaly_results")
    op.drop_table("anomaly_results")
    op.drop_index("ix_detection_runs_algorithm_created_at", table_name="detection_runs")
    op.drop_index("ix_detection_runs_meter_id_created_at", table_name="detection_runs")
    op.drop_index("ix_detection_runs_status_lease_expires_at", table_name="detection_runs")
    op.drop_table("detection_runs")
