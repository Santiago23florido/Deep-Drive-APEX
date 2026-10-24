"""Bound the actual adaptive driver without replacing its steering controller."""
import numpy as np
import pytest

from apex_fusion_research.core.race_config import MissionConfig, RaceDriverConfig
from apex_fusion_research.core.race_driver import DONE, EXPLORE, RaceDriverCore
from apex_fusion_research.core.reactive import ReactiveCommand


def driving_core():
    cfg=RaceDriverConfig()
    cfg.mission=MissionConfig(enabled=True)
    cfg.adaptive.recover_enabled=False
    core=RaceDriverCore(cfg,3)
    core.phase=EXPLORE
    core.mission_started_t=0.0
    core.mission_xy=(0.0,0.0)
    core.local_path_odom=np.column_stack((np.linspace(-.15,3,40),np.zeros(40)))
    core.cmd=ReactiveCommand(0.0,.8,core.local_path_odom,True)
    return core


def test_command_cap_includes_feedback_and_launch_compensation():
    core=driving_core();commands=[]
    for t in np.arange(0,1,.02):
        v,_,_=core.control(float(t),(0.0,0.0,0.0),0.0,None,True)
        commands.append(v)
        assert core.ident.pending[1]==v
    assert max(commands)==pytest.approx(.25)
    assert min(commands)>=0


def test_distance_limit_counts_a_bend_and_latches_stop():
    core=driving_core()
    for t,xy in [(0,(0,0)),(.1,(.6,0)),(.2,(.6,.5))]:
        v,_,_=core.control(t,(*xy,0),.2,None,True)
    assert np.hypot(.6,.5)<core.cfg.mission.max_distance_m
    assert core.mission_distance==pytest.approx(1.1)
    assert v==0 and core.phase==DONE
    assert core.mission_stop_reason=='distance'
    for t in [.3,1,2]:
        assert core.control(t,(.7,.7,0),0,None,True)[0]==0
        assert core.mission_distance==pytest.approx(1.1)


def test_time_limit_stops_even_if_odometry_reports_no_motion():
    core=driving_core()
    assert core.control(0,(0,0,0),0,None,True)[0]>=0
    assert core.control(5,(0,0,0),0,None,True)[0]==0
    assert core.mission_stop_reason=='duration' and core.phase==DONE


def test_reverse_request_aborts_instead_of_driving_backwards():
    core=driving_core();core.cmd.v=-.2
    assert core.control(.02,(0,0,0),0,None,True)[0]==0
    assert core.mission_stop_reason=='reverse requested'
    assert core.phase==DONE


def test_negative_speed_feedback_coasts_without_starting_reverse():
    core=driving_core()
    assert core.control(0,(0,0,0),.2,None,True)[0]==0
    assert core.phase==EXPLORE and core.mission_stop_reason is None


def test_wait_preview_exercises_planner_without_starting_mission_or_motion():
    from apex_fusion_research.core.race_driver import WAIT
    cfg = RaceDriverConfig()
    cfg.reactive.preview_while_waiting = True
    cfg.mission = MissionConfig(enabled=True)
    core = RaceDriverCore(cfg, 1)
    angles = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    for t in np.arange(0, .5, .1):
        core.on_scan(float(t), np.full(360, 2.0), float(angles[0]), float(angles[1]-angles[0]),
                     (0, 0, 0), (0, 0, 0), 0)
        v, _, _ = core.control(float(t), (0, 0, 0), 0, None, False)
        assert v == 0
    assert core.cmd is not None and core.cmd.diag['n_feasible'] > 0
    assert core.local_path_odom is not None and core.react_ms > 0
    assert core.phase == WAIT and core.mission_started_t is None


def test_default_wait_keeps_planning_preview_disabled():
    core = RaceDriverCore(RaceDriverConfig(), 1)
    core.on_scan(0, np.full(360, 2.0), -np.pi, np.pi/180, (0, 0, 0), (0, 0, 0), 0)
    assert core.cmd is None


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf')])
@pytest.mark.parametrize('field',['max_distance_m','max_duration_s','max_command_mps','max_reverse_command_mps'])
def test_invalid_limits_are_rejected(field,value):
    with pytest.raises(ValueError):MissionConfig(**{field:value})
