export function highlightNodeAndEdges(graphData, svg, selectedNode) {
    const graphContainer = svg.select(".graph-container");

    if (!selectedNode || !graphData) return;

    const nodes = graphContainer.selectAll("circle");
    const edges = graphContainer.selectAll("path");  // Changed from 'line' to 'path'

    const resetToDefault = () => {
        nodes.attr('fill', d => {
            if (!d) return '#999999';
            if (d.connections > 4) return '#32CD32';
            if (d.connections > 2) return '#32CD32';
            return '#999999';
        })
        .attr('r', d => {
            if (!d) return 5;
            const baseRadius = 10 + (d.importance * 50);
            const connectionRadius = d.connections * 3;
            return Math.min(baseRadius + connectionRadius, 50);
        })
        .style('opacity', 1)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .classed("highlighted", false);

        edges.attr('stroke', '#999')
            .attr('stroke-width', 1)
            .style('opacity', 0.6);
    };

    const isHighlighted = selectedNode.highlighted;

    if (isHighlighted) {
        resetToDefault();
        selectedNode.highlighted = false;
        return;
    }

    resetToDefault();

    const connectedEdges = graphData.edges.filter(edge => {
        const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
        const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
        return sourceId === selectedNode.id || targetId === selectedNode.id;
    });

    edges.style('opacity', d => {
        const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
        const targetId = typeof d.target === 'object' ? d.target.id : d.target;
        const isConnected = connectedEdges.some(edge => {
            const edgeSourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
            const edgeTargetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
            return (sourceId === edgeSourceId && targetId === edgeTargetId) ||
                   (sourceId === edgeTargetId && targetId === edgeSourceId);
        });
        return isConnected ? 1 : 0.1;  // Made non-connected edges more transparent
    })
    .attr('stroke', d => {
        const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
        const targetId = typeof d.target === 'object' ? d.target.id : d.target;
        const isConnected = connectedEdges.some(edge => {
            const edgeSourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
            const edgeTargetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
            return (sourceId === edgeSourceId && targetId === edgeTargetId) ||
                   (sourceId === edgeTargetId && targetId === edgeSourceId);
        });
        return isConnected ? '#000000' : '#999';
    })
    .attr('stroke-width', d => {
        const sourceId = typeof d.source === 'object' ? d.source.id : d.source;
        const targetId = typeof d.target === 'object' ? d.target.id : d.target;
        const isConnected = connectedEdges.some(edge => {
            const edgeSourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
            const edgeTargetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
            return (sourceId === edgeSourceId && targetId === edgeTargetId) ||
                   (sourceId === edgeTargetId && targetId === edgeSourceId);
        });
        return isConnected ? 2 : 1;
    });

    const connectedNodeIds = new Set(connectedEdges.flatMap(edge => {
        const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
        const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
        return [sourceId, targetId];
    }));

    nodes
        .style('opacity', d => d.id === selectedNode.id || connectedNodeIds.has(d.id) ? 1 : 0.2)
        .attr('stroke', d => (d.id === selectedNode.id || connectedNodeIds.has(d.id)) ? '#000000' : '#fff')
        .attr('stroke-width', d => (d.id === selectedNode.id || connectedNodeIds.has(d.id)) ? 3 : 1.5)
        .classed("highlighted", d => d.id === selectedNode.id || connectedNodeIds.has(d.id));

    selectedNode.highlighted = true;
}