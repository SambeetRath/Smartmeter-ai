"""Create the initial meter and meter-reading schema.

Revision ID: 20261030_0001
Revises:
Create Date: 2026-10-30
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "20261030_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "meters",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("meter_identifier", sa.String(length=100), nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_meters"),
        sa.UniqueConstraint("meter_identifier", name="uq_meters_meter_identifier"),
    )
    op.create_table(
        "meter_readings",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("meter_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("global_active_power", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("global_reactive_power", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("voltage", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("global_intensity", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("sub_metering_1", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("sub_metering_2", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.Column("sub_metering_3", sa.Numeric(precision=12, scale=6), nullable=True),
        sa.ForeignKeyConstraint(
            ["meter_id"],
            ["meters.id"],
            name="fk_meter_readings_meter_id_meters",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_meter_readings"),
        sa.UniqueConstraint(
            "meter_id",
            "timestamp",
            name="uq_meter_readings_meter_timestamp",
        ),
    )


def downgrade() -> None:
    op.drop_table("meter_readings")
    op.drop_table("meters")
