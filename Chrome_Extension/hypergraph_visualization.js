// hypergraph_visualization.js
// Hypergraph 데이터 로드
let selectedEdgeId = null; // 선택된 엣지 ID 추적용

function visualizeHypergraph(data) {
    const svg = d3.select("svg");

    const margin = { right: 70 }; // 오른쪽에 범례를 위한 여백
    const width = svg.node().getBoundingClientRect().width - margin.right;
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
        .velocityDecay(0.5)
        .force("charge", d3.forceCollide().radius(50))
        .force("link", d3.forceLink(data.edges.flatMap(edge => 
            edge.nodes.length === 2 ? [{
                source: data.nodes.find(n => n.id === edge.nodes[0]),
                target: data.nodes.find(n => n.id === edge.nodes[1])
            }] : edge.nodes.map((nodeId, index) => ({
                source: data.nodes.find(n => n.id === nodeId),
                target: data.nodes.find(n => n.id === edge.nodes[(index + 1) % edge.nodes.length])
            }))
        )).distance(10))
        .force("center", d3.forceCenter((width - margin.right) / 2, height / 2))
        .force("radial", d3.forceRadial(MAX_DISTANCE, (width - margin.right) / 2, height / 2))
        .force("boundary", () => {
            data.nodes.forEach(node => {
                if (node !== centralNode) {
                    node.x = Math.max(50, Math.min(width - 50, node.x));
                    node.y = Math.max(50, Math.min(height - 50, node.y));
                }
            });
        });

    // 중앙 노드 위치 고정
    centralNode.fx = (width - margin.right) / 2;
    centralNode.fy = height / 2;

    // 하이퍼엣지 연결선 (하이퍼엣지로 이루어진 노드들의 중앙으로 연결)
    let edgeLines = svg.append("g")
        .attr("class", "edges")
        .selectAll("line")
        .data(data.edges.flatMap(d => {
            if (d.nodes.length === 2) {
                return [{ edge: d, nodeId1: d.nodes[0], nodeId2: d.nodes[1] }];
            } else {
                return d.nodes.map(nodeId => ({ edge: d, nodeId: nodeId }));
            }
        }))
        .enter()
        .append("line")
        .attr("stroke", d => color(d.edge.category))
        .attr("stroke-width", 1)
        .style("cursor", "pointer")
        .on("click", function(event, d) {
            const edgeId = d.edge.id;
            const url = `edge_info.html?edgeId=${edgeId}`;
            window.open(url, '_blank');
            createEdgeInfoTooltip(event, d);
        });

    edgeLines
        .on("mouseover", function(event, d) {
            const tooltip = d3.select("body")
                .append("div")
                .attr("class", "edge-tooltip")
                .style("position", "absolute")
                .style("background", "white")
                .style("border", "1px solid black")
                .style("padding", "5px")
                .style("border-radius", "3px")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
            
            tooltip.append("div")
                .text(`${d.edge.category}`);
        })
        .on("mouseout", function() {
            d3.selectAll(".edge-tooltip").remove();
        });

    // 노드 그룹 컨테이너 생성
    let nodeGroup = svg.append("g")
        .attr("class", "nodes")
        .selectAll("g")
        .data(data.nodes)
        .enter().append("g");

    // 하이퍼엣지 경계박스 생성 함수
    function createHyperedgeBoundingBox(edge) {
        // 해당 엣지의 노드들 찾기
        const nodeCircles = nodeGroup.append("circle")
            .attr("r", d => {
                // 중앙 노드인 경우 더 큰 크기 적용
                if (d === centralNode) {
                    return Math.max((d.importance || 1) * 30, 20);  // 중앙 노드는 30배, 최소 20
                }
                // 일반 노드
                return Math.max((d.importance || 1) * 15, 10);  // 일반 노드는 15배, 최소 10
            })
            .attr("fill", "#32CD32")
            .attr("class", "node-circle");

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
            .attr("stroke-width", 1)
            .attr("stroke-dasharray", "10,5");

        return boundingBox;
    }

    // 노드 (원)
    const nodeCircles = nodeGroup.append("circle")
        .attr("r", d => d === centralNode 
            ? Math.max((d.importance || 1) * 30, 25)  // 중앙 노드는 항상 큼
            : Math.max((d.importance || 1) * 15, 10)  // 일반 노드는 더 작음
        )
        .attr("fill", "#32CD32")
        .attr("class", "node-circle");

    // 노드 레이블 (제목)
    const nodeLabels = nodeGroup.append("text")
        .text(d => d.id)
        .attr("fill", "black")
        .attr("dy", d => d === centralNode ? 5 : -20) 
        .attr("text-anchor", "middle")
        .style("pointer-events", "none")
        .style("font-size", d => d === centralNode ? "14px" : "12px")
        .attr("font-weight", d => d === centralNode ? "bold" : "normal")
        .style("font-family", "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif");

    // 카테고리 중복 제거
    const uniqueCategories = Array.from(new Set(data.edges.map(edge => edge.category)));
    
    // 범례 부분 수정
    const legend = svg.append("g")
    .attr("class", "legend")
    .attr("transform", `translate(${width - margin.right + 20}, 20)`);

    // View All 버튼을 먼저 생성 (맨 위에)
    const resetLegendItem = legend.append("g")
    .attr("transform", `translate(0, 0)`)  // 위치를 맨 위로
    .attr("class", "legend-item")
    .style("cursor", "pointer")
    .on("click", () => {
        // 모든 노드와 엣지를 원래 상태로 복원
        nodeGroup.style("opacity", 1);
        edgeLines.style("opacity", 1);
    });

    resetLegendItem.append("rect")
    .attr("x", -5)
    .attr("y", -15)
    .attr("width", 130)
    .attr("height", 25)
    .attr("fill", "#f5f5f5")  // 배경색만 다르게
    .attr("stroke", "#ddd")
    .attr("rx", 3)
    .style("opacity", 0.8);

    resetLegendItem.append("circle")
    .attr("r", 6)
    .attr("fill", "gray")
    .attr("cx", 10)
    .attr("cy", 0);

    resetLegendItem.append("text")
    .attr("x", 25)
    .attr("y", 4)
    .text("View All")
    .style("font-size", "12px")
    .style("font-weight", "500");

    // 호버 효과 추가
    resetLegendItem
    .on("mouseover", function() {
        d3.select(this).select("rect")
            .transition()
            .duration(200)
            .style("opacity", 1);
    })
    .on("mouseout", function() {
        d3.select(this).select("rect")
            .transition()
            .duration(200)
            .style("opacity", 0.8);
    });

    // 카테고리별 범례 아이템 (View All 다음부터)
    uniqueCategories.forEach((category, index) => {
    const legendItem = legend.append("g")
        .attr("transform", `translate(0, ${(index + 1) * 30})`)  // index + 1로 위치 조정
        .attr("class", "legend-item")
        .style("cursor", "pointer")
        .on("click", () => {
            // 현재 선택된 카테고리의 엣지들 찾기
            const categoryEdges = data.edges.filter(e => e.category === category);
            const visibleNodeIds = new Set(categoryEdges.flatMap(edge => edge.nodes));
        
            // 노드 필터링
            nodeGroup.style("opacity", d => {
                // 중앙 노드는 항상 표시
                if (d === centralNode) return 1;
                return visibleNodeIds.has(d.id) ? 1 : 0.2;
            });
        
            // 엣지 필터링
            edgeLines.style("opacity", d => 
                categoryEdges.some(edge => edge.id === d.edge.id) ? 1 : 0.2
            );
        });

    legendItem.append("rect")
        .attr("x", -5)
        .attr("y", -15)
        .attr("width", 130)
        .attr("height", 25)
        .attr("fill", "white")
        .attr("stroke", "#ddd")
        .attr("rx", 3)
        .style("opacity", 0.8);

    legendItem.append("circle")
        .attr("r", 6)
        .attr("fill", color(category))
        .attr("cx", 10)
        .attr("cy", 0);

    legendItem.append("text")
        .attr("x", 25)
        .attr("y", 4)
        .text(category)
        .style("font-size", "12px")
        .style("font-weight", "500");

    // 호버 효과
    legendItem
        .on("mouseover", function() {
            d3.select(this).select("rect")
                .transition()
                .duration(200)
                .style("opacity", 1);
        })
        .on("mouseout", function() {
            d3.select(this).select("rect")
                .transition()
                .duration(200)
                .style("opacity", 0.8);
        });
    });

    // 마우스 오버/아웃 이벤트
    nodeGroup
        .on("mouseover", function(event, d) {
            const connectedEdges = data.edges.filter(edge => edge.nodes.includes(d.id));

            // 노드 크기 변경
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", node => {
                    if (node === centralNode) {
                        return node === d ? Math.max((node.importance || 1) * 35, 30) : Math.max((node.importance || 1) * 30, 25);
                    }
                    if (node.id === d.id) {
                        return Math.max((node.importance || 1) * 20, 15);
                    }
                    return Math.max((node.importance || 1) * 15, 10);
                });

            // 텍스트 크기 변경
            nodeGroup.selectAll("text")
                .transition()
                .duration(200)
                .style("font-size", node => {
                    if (node === centralNode) {
                        return node === d ? "18px" : "16px";
                    }
                    if (node.id === d.id) {
                        return "14px";
                    }
                    return "12px";
                });

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", edge => {
                    if (connectedEdges.includes(edge.edge)) return 2;
                    return 1;
                })
                .attr("stroke", "gray");
        })
        .on("mouseout", function() {
            // 노드 크기 원복
            nodeGroup.selectAll(".node-circle")
                .transition()
                .duration(200)
                .attr("r", d => d === centralNode 
                    ? Math.max((d.importance || 1) * 30, 25)  
                    : Math.max((d.importance || 1) * 15, 10)  
                );

            // 텍스트 크기 원복
            nodeGroup.selectAll("text")
                .transition()
                .duration(200)
                .style("font-size", d => d === centralNode ? "16px" : "12px");

            edgeLines
                .transition()
                .duration(200)
                .attr("stroke-width", 1)
                .attr("stroke", d => color(d.edge.category));
        });

    // 드래그 이벤트 설정
    const drag = d3.drag()
        .on("start", dragStarted)
        .on("drag", dragged)
        .on("end", dragEnded);

    nodeGroup.call(drag);

    function dragStarted(event, d) {
        if (d === centralNode) return;
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }

    function dragged(event, d) {
        if (d === centralNode) return;
        const dx = event.x - width / 2;
        const dy = event.y - height / 2;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance >= MIN_DISTANCE && distance <= MAX_DISTANCE) {
            d.fx = Math.max(0.05 * width, Math.min(0.95 * width, event.x));
            d.fy = Math.max(0.05 * height, Math.min(0.95 * height, event.y));
        }
    }

    function dragEnded(event, d) {
        if (d === centralNode) return;
        if (!event.active) simulation.alphaTarget(0);
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

    // Edge info tooltip function
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

    // Click event to close tooltip
    d3.select("body").on("click", function(event) {
        if (!event.target.closest(".edge-tooltip")) {
            d3.selectAll(".edge-tooltip").remove();
        }
    });
}