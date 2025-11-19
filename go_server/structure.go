package main

import "strings"

type ModelConfig struct {
	ID         string  `json:"id"`
	SLO        float64 `json:"SLO"`
	StartStage string  `json:"start_stage"`
}

type Stage struct {
	ID     string   `json:"id"`
	Models []string `json:"models"`
}

type EdgeConfig struct {
	Source string   `json:"source"`
	Target string   `json:"target"`
	Models []string `json:"models"`
	Type   string   `json:"type"`
}

type DAGConfig struct {
	Stages []Stage      `json:"stages"`
	Edges  []EdgeConfig `json:"edges"`
}

type StageProfile struct {
	ID             string    `json:"id"`
	Latency        float64   `json:"latency"`
	MaxBS          int32     `json:"max_bs"`
	NumOutput      uint32    `json:"num_output"`
	ShapeInput     [][]int32 `json:"shape_input"`
	ModelSize      int64     `json:"model_size"`
	InputSize      int64     `json:"input_size"`
	RunningMaxSize int64     `json:"running_max_size"`
	FillTimeOut    float64   `json:"fill_time_out"`
}

type SchedulerConfig struct {
	ModelConfig      []ModelConfig  `json:"model_config"`
	DAGConfig        DAGConfig      `json:"dag_config"`
	StageProfile     []StageProfile `json:"stage_profile"`
	DeviceNum        int32          `json:"device_num"`
	StaticDeployment [][]string     `json:"static_deployment"`
	JustStatic       bool           `json:"just_static"`
	Discard          bool           `json:"discard"`
	Debug            bool           `json:"debug"`
	DebugTimeMaxSize int            `json:"debug_size"`
}

type ServerConfig struct {
	ServerAddr  string `json:"server_addr"`
	ServerPort  string `json:"server_port"`
	MaxDataSize int    `json:"max_data_size"`
}

type PredictRequest struct {
	ModelID string `json:"model_id"`
	Input   []byte `json:"input,omitempty"`
}

// DAG

type EdgeType int

const (
	Normal EdgeType = iota
	Direct
)

type Edge struct {
	Target string
	Type   EdgeType
}

// DAG 结构体
type DAG struct {
	Stages map[string]*Stage           `json:"stages"`
	Edges  map[string]map[string]*Edge `json:"edges"`
}

// NewDAG 创建新的 DAG
func NewDAG() *DAG {
	return &DAG{
		Stages: make(map[string]*Stage),
		Edges:  make(map[string]map[string]*Edge), // stageID -> (modelID -> nextStageID)
	}
}

// AddStage 添加一个 Stage
func (d *DAG) AddStage(stage *Stage) {
	if _, exists := d.Stages[stage.ID]; !exists {
		d.Stages[stage.ID] = stage
		d.Edges[stage.ID] = make(map[string]*Edge)
	}
}

// AddEdge 添加一条边（从 start 连接到 end 通过 model）
func (d *DAG) AddEdge(start, end, model string, _type string) {
	if _, exists := d.Edges[start]; !exists {
		d.Edges[start] = make(map[string]*Edge)
	}
	var eType EdgeType
	if strings.EqualFold(_type, "Direct") {
		eType = Direct
	} else {
		eType = Normal
	}
	d.Edges[start][model] = &Edge{
		Target: end,
		Type:   eType,
	}
}

func (d *DAG) NextStageWithThis(stageID string, modelID string) []string {
	ans := []string{stageID}
	for {
		edges, ok := d.Edges[stageID]
		if !ok {
			break
		}
		edge, ok := edges[modelID]
		if ok && edge.Type == Direct {
			stageID = edge.Target
			ans = append(ans, stageID)
		} else {
			break
		}
	}
	return ans
}

func (d *DAG) NextStage(stageID string, modelID string) []string {
	edges, ok := d.Edges[stageID]
	if !ok {
		return []string{}
	}
	edge, ok := edges[modelID]
	if !ok {
		return []string{}
	}
	return d.NextStageWithThis(edge.Target, modelID)
}

func NewDAGFromConfig(dagConfig *DAGConfig) *DAG {
	d := NewDAG()
	for _, s := range dagConfig.Stages {
		d.AddStage(&s)
	}
	for _, e := range dagConfig.Edges {
		for _, m := range e.Models {
			d.AddEdge(e.Source, e.Target, m, e.Type)
		}
	}
	return d
}
