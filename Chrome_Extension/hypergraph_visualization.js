// hypergraph_visualization.js
// Hypergraph 데이터 로드
function visualizeHypergraph (data) {
    const svg = d3.select("svg");
    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    svg.selectAll("*").remove();

    const color = d3.scaleOrdinal(d3.schemeTableau10);

    // 가장 중요도가 큰 노드 찾기
    const centralNode = data.nodes.reduce((maxNode, node) => (node.importance || 0) > (maxNode.importance || 0) ? node : maxNode, data.nodes[0]);

    // 중심 노드로부터의 최소/최대 거리 제한
    const MIN_DISTANCE = 100;  // 중심으로부터 최소 거리
    const MAX_DISTANCE = 300;  // 중심으로부터 최대 거리

    // Force Simulation 설정
    const simulation = d3.forceSimulation(data.nodes)
        .force("charge", d3.forceCollide().radius(50))  // 노드 간 충돌 방지
        .force("link", d3.forceLink(data.edges.flatMap(edge => 
            edge.nodes.length === 2 ? [{
                source: data.nodes.find(n => n.id === edge.nodes[0]),
                target: data.nodes.find(n => n.id === edge.nodes[1])
            }] : edge.nodes.map((nodeId, index) => ({
                source: data.nodes.find(n => n.id === nodeId),
                target: data.nodes.find(n => n.id === edge.nodes[(index + 1) % edge.nodes.length])
            }))
        )).distance(100))  // 노드 간 거리 설정
        .force("center", d3.forceCenter(width / 2, height / 2))  // 중앙 배치
        .force("radial", d3.forceRadial(MAX_DISTANCE, width / 2, height / 2));  // 원형 배치

    // 중앙 노드 위치 고정
    centralNode.fx = width / 2;
    centralNode.fy = height / 2;

    // 노드 그룹 컨테이너 생성
    let nodeGroup = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(data.nodes)
        .enter().append("g");

    // 하이퍼엣지 연결선 (하이퍼엣지로 이루어진 노드들의 중앙으로 연결)
    let edgeLines = svg.append("g")
    .attr("class", "edges")
    .selectAll("line")
    .data(data.edges.flatMap(d => {
        if (d.nodes.length === 2) {
            // 노드 2개만 연결된 엣지는 직선으로 이어지도록 설정
            return [{ edge: d, nodeId1: d.nodes[0], nodeId2: d.nodes[1] }];
        } else {
            return d.nodes.map(nodeId => ({ edge: d, nodeId: nodeId }));
        }
    }))
    .enter()
    .append("line")
    .attr("stroke", (d, i) => color(d.edge.id))
    .attr("stroke-width", d => (d.edge.importance || 1) * 3)
    .on("click", function(event, d) {
        // 새 창 열기
        const edgeId = d.edge.id;
        const url = `edge_info.html?edgeId=${edgeId}`;
        window.open(url, '_blank');
        
        // 툴팁 생성
        createEdgeInfoTooltip(event, d);
    });

    // 하이퍼엣지 경계박스 생성 함수
    function createHyperedgeBoundingBox(edge) {
        // 해당 엣지의 노드들 찾기
        const edgeNodes = edge.nodes.map(nodeId => 
            data.nodes.find(node => node.id === nodeId)
        );

        // 노드들의 좌표 계산
        const xCoords = edgeNodes.map(node => node.x);
        const yCoords = edgeNodes.map(node => node.y);

        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);

        // 여유 공간 추가
        const padding = 30;

        // 경계 박스 생성
        const boundingBox = svg.append("rect")
            .attr("class", "hyperedge-bounding-box")
            .attr("x", minX - padding)
            .attr("y", minY - padding)
            .attr("width", maxX - minX + 2 * padding)
            .attr("height", maxY - minY + 2 * padding)
            .attr("fill", "none")
            .attr("stroke", color(edge.id))
            .attr("stroke-width", 3)
            .attr("stroke-dasharray", "10,5");

        return boundingBox;
    }

    // 노드 (원)
    const nodeCircles = nodeGroup.append("circle")
        .attr("r", d => Math.max((d.importance || 1) * 15, 10))
        .attr("fill", "gray")
        .attr("class", "node-circle");

    // 노드 레이블 (제목)
    nodeGroup.append("text")
        .text(d => d.id)
        .attr("fill", "black")
        .attr("dy", -20)
        .style("pointer-events", "none");

    // 엣지 필터링 함수
    function updateVisualizationForEdge(visibleEdgeId) {
        if (visibleEdgeId === null) {
            // 모든 노드와 엣지를 표시
            nodeGroup.style("opacity", 1);
            edgeLines.style("opacity", 1);
        } else {
            // 특정 엣지와 연결된 노드만 표시
            const filteredEdges = data.edges.filter(edge => edge.id === visibleEdgeId);
            const visibleNodeIds = new Set(filteredEdges.flatMap(edge => edge.nodes));

            // 노드 필터링
            nodeGroup.style("opacity", d => visibleNodeIds.has(d.id) ? 1 : 0.2);

            // 엣지 필터링
            edgeLines.style("opacity", edge => 
                filteredEdges.some(filteredEdge => filteredEdge.id === edge.edge.id) ? 1 : 0.2
            );
        }
    }

    // 범례 생성
    const legend = svg.append("g")
        .attr("class", "legend")
        .attr("transform", `translate(${width - 150}, 20)`);

    const uniqueEdges = Array.from(new Set(data.edges.map(edge => edge.id)));
    
    // 범례 아이템 생성
    uniqueEdges.forEach((edgeId, index) => {
        const edge = data.edges.find(e => e.id === edgeId);
        
        const legendItem = legend.append("g")
            .attr("transform", `translate(0, ${index * 25})`)
            .attr("class", "legend-item")
            .style("cursor", "pointer")
            .on("click", () => {
                // 기존 경계 박스 제거
                svg.selectAll(".hyperedge-bounding-box").remove();

                // 엣지 필터링
                updateVisualizationForEdge(edgeId);

                // 3개 이상의 노드를 가진 하이퍼엣지인 경우 경계 박스 생성
                if (edge.nodes.length >= 3) {
                    createHyperedgeBoundingBox(edge);
                }

                // 설명 패널 생성
                const descriptionPanel = svg.append("g")
                    .attr("class", "edge-description-panel")
                    .attr("transform", `translate(${width - 300}, ${height - 150})`);
                
                descriptionPanel.append("rect")
                    .attr("width", 250)
                    .attr("height", 100)
                    .attr("fill", "white")
                    .attr("stroke", color(edgeId));
                
                descriptionPanel.append("text")
                    .attr("x", 10)
                    .attr("y", 20)
                    .attr("font-weight", "bold")
                    .text(`Edge ${edgeId} Description`);
                
                descriptionPanel.append("text")
                    .attr("x", 10)
                    .attr("y", 40)
                    .attr("width", 230)
                    .attr("font-size", "0.8em")
                    .text(data.edges.find(e => e.id === edgeId)?.description || 'No description available.')
            });

        legendItem.append("circle")
            .attr("r", 8)
            .attr("fill", color(edgeId))
            .attr("cx", 10)
            .attr("cy", 10);

        legendItem.append("text")
            .attr("x", 40)
            .attr("y", 15)
            .text(`Edge ${edgeId}`);
    });

    // 전체 그래프 보기 클릭 원 추가
    const resetLegendItem = legend.append("g")
        .attr("transform", `translate(0, ${uniqueEdges.length * 25})`)
        .attr("class", "legend-item")
        .style("cursor", "pointer")
        .on("click", () => {
            // 경계 박스 제거
            svg.selectAll(".hyperedge-bounding-box").remove();

            // 전체 그래프를 다시 그림
            updateVisualizationForEdge(null);
        });

    resetLegendItem.append("circle")
        .attr("r", 8)
        .attr("fill", "grey")
        .attr("cx", 10)
        .attr("cy", 10);

    resetLegendItem.append("text")
        .attr("x", 40)
        .attr("y", 15)
        .text("View All");

    // 마우스 오버/아웃 이벤트
    nodeGroup
        .on("mouseover", function(event, d) {
            const connectedEdges = data.edges.filter(edge => edge.nodes.includes(d.id));

            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", node => {
                    if (node.id === d.id) return Math.max((node.importance || 1) * 25, 20);
                    return Math.max((node.importance || 1) * 15, 10);
                })
                .attr("fill", node => node.id === d.id ? "red" : "gray");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", edge => {
                    if (connectedEdges.includes(edge.edge)) return (edge.edge.importance || 1) * 6;
                    return (edge.edge.importance || 1) * 3;
                })
                .attr("stroke", edge => {
                    if (connectedEdges.includes(edge.edge)) return "red";
                    return color(edge.edge.id);
                });
        })
        .on("mouseout", function() {
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", d => Math.max((d.importance || 1) * 15, 10))
                .attr("fill", "gray");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", d => (d.edge.importance || 1) * 3)
                .attr("stroke", d => color(d.edge.id));
        });

    // 드래그 이벤트 설정
    const drag = d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);

    nodeGroup.call(drag);

    function dragStarted(event, d) {
        // 중앙 노드는 드래그 불가
        if (d === centralNode) return;

        // 시뮬레이션 일시 중지
        if (!event.active) simulation.alphaTarget(0.3).restart();
        
        // 노드 위치 고정
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        // 중앙 노드는 드래그 불가
        if (d === centralNode) return;

        // 중심 노드로부터의 거리 제한
        const dx = event.x - width / 2;
        const dy = event.y - height / 2;
        const distance = Math.sqrt(dx * dx + dy * dy);

        // 최소 및 최대 거리 제한
        if (distance >= MIN_DISTANCE && distance <= MAX_DISTANCE) {
            // 화면 경계 내로 제한
            d.fx = Math.max(0.05 * width, Math.min(0.95 * width, event.x));
            d.fy = Math.max(0.05 * height, Math.min(0.95 * height, event.y));
        }
    }

    function dragEnded(event, d) {
        // 중앙 노드는 드래그 불가
        if (d === centralNode) return;

        // 시뮬레이션 재개
        if (!event.active) simulation.alphaTarget(0);

        // 드래그 종료 시 위치 고정 해제 (자유롭게 움직일 수 있도록)
        d.fx = null;
        d.fy = null;
    }

    // 위치 업데이트 및 시뮬레이션 이벤트
    simulation.on("tick", () => {
        nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);

        edgeLines
            .attr("x1", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node1 = data.nodes.find(n => n.id === d.nodeId1);
                    return node1.x;
                } else {
                    const edgeNodes = d.edge.nodes.map(nodeId => data.nodes.find(n => n.id === nodeId));
                    return d3.mean(edgeNodes, node => node.x);
                }
            })
            .attr("y1", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node1 = data.nodes.find(n => n.id === d.nodeId1);
                    return node1.y;
                } else {
                    const edgeNodes = d.edge.nodes.map(nodeId => data.nodes.find(n => n.id === nodeId));
                    return d3.mean(edgeNodes, node => node.y);
                }
            })
            .attr("x2", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node2 = data.nodes.find(n => n.id === d.nodeId2);
                    return node2.x;
                } else {
                    const node = data.nodes.find(n => n.id === d.nodeId);
                    return node.x;
                }
            })
            .attr("y2", d => {
                if (d.nodeId1 && d.nodeId2) {
                    const node2 = data.nodes.find(n => n.id === d.nodeId2);
                    return node2.y;
                } else {
                    const node = data.nodes.find(n => n.id === d.nodeId);
                    return node.y;
                }
            });
    });

    // Edge descriptions object
    const edgeDescriptions = {
        '1': 'This edge represents a key relationship between connected nodes, highlighting their interconnectedness.',
        '2': 'A complex interconnection showcasing the multifaceted nature of the nodes involved.',
        '3': 'An important structural connection that bridges multiple nodes in the hypergraph.'
    };

    // Modify the existing code to include edge descriptions
    function createEdgeInfoTooltip(event, d) {
        const edgeId = d.edge.id;
        const description = d.edge.description || 'No additional description available.';
        
        const tooltip = d3.select("body")
            .append("div")
            .attr("class", "edge-tooltip")
            .style("position", "absolute")
            .style("background", "white")
            .style("border", "1px solid black")
            .style("padding", "10px")
            .style("border-radius", "5px")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px");
        
        tooltip.append("h3")
            .text(`Edge ${edgeId} Details`);
        
        tooltip.append("p")
            .text(description);
        
        tooltip.append("div")
            .text(`Nodes: ${d.edge.nodes.join(", ")}`)
            .style("font-size", "0.8em")
            .style("color", "gray");
    }

    // Optional: Add a way to close the tooltip
    d3.select("body").on("click", function(event) {
        if (!event.target.closest(".edge-tooltip")) {
            d3.selectAll(".edge-tooltip").remove();
        }
    });

}