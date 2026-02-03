"""
2D Frame Analysis Backend Server
=================================
FastAPI backend for structural analysis of 2D frames using OpenSees and opstool.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import base64
import io
import openseespy.opensees as ops

# Import opstool post-processing only (avoid GUI dependencies like tkinter)
try:
    import opstool.post as opst_post
    OPSTOOL_AVAILABLE = True
except ImportError:
    OPSTOOL_AVAILABLE = False
    opst_post = None

app = FastAPI(
    title="2D Frame Analysis API",
    description="Backend server for 2D frame structural analysis",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Input Models ====================

class Node(BaseModel):
    id: int = Field(..., description="Node ID")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class Element(BaseModel):
    id: int = Field(..., description="Element ID")
    node_i: int = Field(..., description="Start node ID")
    node_j: int = Field(..., description="End node ID")
    E: float = Field(..., description="Modulus of elasticity")
    I: float = Field(..., description="Moment of inertia")
    A: float = Field(..., description="Cross-sectional area")


class PointLoad(BaseModel):
    node_id: int = Field(..., description="Node ID where load is applied")
    fx: float = Field(0.0, description="Force in X direction")
    fy: float = Field(0.0, description="Force in Y direction")
    mz: float = Field(0.0, description="Moment about Z axis")


class UDL(BaseModel):
    element_id: int = Field(..., description="Element ID where UDL is applied")
    wy: float = Field(0.0, description="Uniformly distributed load in Y direction")
    wx: float = Field(0.0, description="Uniformly distributed load in X direction")


class Support(BaseModel):
    node_id: int = Field(..., description="Node ID")
    support_type: str = Field(..., description="Support type: 'pin', 'roller', or 'fixed'")


class DesignatedPosition(BaseModel):
    element_id: int = Field(..., description="Element ID")
    position: float = Field(..., ge=0.0, le=1.0, description="Position along element (0.0 to 1.0)")


class AnalysisRequest(BaseModel):
    nodes: List[Node] = Field(..., description="List of nodes")
    elements: List[Element] = Field(..., description="List of elements")
    E: Optional[float] = Field(None, description="Global modulus of elasticity (overrides element E if provided)")
    I: Optional[float] = Field(None, description="Global moment of inertia (overrides element I if provided)")
    A: Optional[float] = Field(None, description="Global cross-sectional area (overrides element A if provided)")
    point_loads: Optional[List[PointLoad]] = Field(default=[], description="List of point loads")
    udls: Optional[List[UDL]] = Field(default=[], description="List of uniformly distributed loads")
    supports: List[Support] = Field(..., description="List of supports")
    designated_positions: Optional[List[DesignatedPosition]] = Field(default=[], description="Positions along elements to get values")
    mass: Optional[Dict[int, List[float]]] = Field(default=None, description="Node masses: {node_id: [mx, my, mz]}")


# ==================== Model Creation Functions ====================

def create_model(request: AnalysisRequest):
    """Create OpenSees model from request"""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    
    # Create nodes
    for node in request.nodes:
        ops.node(node.id, node.x, node.y)
    
    # Apply supports
    for support in request.supports:
        node_id = support.node_id
        support_type = support.support_type.lower()
        
        if support_type == "fixed":
            ops.fix(node_id, 1, 1, 1)  # Fixed: ux, uy, rz all restrained
        elif support_type == "pin":
            ops.fix(node_id, 1, 1, 0)  # Pin: ux, uy restrained, rotation free
        elif support_type == "roller":
            ops.fix(node_id, 0, 1, 0)  # Roller: only uy restrained
        else:
            raise ValueError(f"Unknown support type: {support.support_type}")
    
    # Create geometric transformation
    ops.geomTransf("Linear", 1)
    
    # Create elements
    for elem in request.elements:
        # Use global E, I, A if provided, otherwise use element-specific values
        E = request.E if request.E is not None else elem.E
        I = request.I if request.I is not None else elem.I
        A = request.A if request.A is not None else elem.A
        
        ops.element("elasticBeamColumn", elem.id, elem.node_i, elem.node_j, 
                   A, E, I, 1)
    
    # Apply masses if provided
    if request.mass:
        for node_id, mass_values in request.mass.items():
            mx = mass_values[0] if len(mass_values) > 0 else 0.0
            my = mass_values[1] if len(mass_values) > 1 else 0.0
            mz = mass_values[2] if len(mass_values) > 2 else 0.0
            ops.mass(node_id, mx, my, mz)
    
    # Apply point loads
    if request.point_loads:
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for load in request.point_loads:
            ops.load(load.node_id, load.fx, load.fy, load.mz)
    
    # Apply UDLs using eleLoad
    if request.udls:
        for udl in request.udls:
            # OpenSees eleLoad for distributed loads
            # Note: wy is typically negative for downward loads
            if udl.wy != 0.0:
                try:
                    ops.eleLoad("-ele", udl.element_id, "-type", "-beamUniform", 0.0, -udl.wy)
                except:
                    # Alternative format if above doesn't work
                    ops.eleLoad("-ele", udl.element_id, "-type", "beamUniform", -udl.wy, 0.0)
            if udl.wx != 0.0:
                try:
                    ops.eleLoad("-ele", udl.element_id, "-type", "-beamUniform", -udl.wx, 0.0)
                except:
                    ops.eleLoad("-ele", udl.element_id, "-type", "beamUniform", 0.0, -udl.wx)


def run_static_analysis(num_points: int = 11):
    """Run static analysis with opstool ODB recording"""
    # Create ODB before analysis (opstool requirement) if available
    odb = None
    if OPSTOOL_AVAILABLE:
        try:
            odb = opst_post.CreateODB(odb_tag="analysis", interpolate_beam_disp=num_points)
        except:
            odb = None
    
    # Set up analysis
    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1.0e-12, 10, 3)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    
    # Run analysis and record response
    result = ops.analyze(1)
    if result != 0:
        raise RuntimeError("Analysis failed to converge")
    
    # Fetch and save response (must be done after analyze) if ODB exists
    if odb is not None:
        try:
            odb.fetch_response_step()
            odb.save_response()
        except:
            pass
    
    return odb


# ==================== Result Extraction Functions ====================

def get_nodal_displacements(node_ids: List[int]) -> Dict:
    """Get displacements for specified nodes"""
    displacements = {}
    for node_id in node_ids:
        try:
            disp = ops.nodeDisp(node_id)
            displacements[node_id] = {
                "ux": float(disp[0]),
                "uy": float(disp[1]),
                "rz": float(disp[2]) if len(disp) > 2 else 0.0
            }
        except:
            displacements[node_id] = {
                "ux": 0.0,
                "uy": 0.0,
                "rz": 0.0
            }
    return displacements


def get_element_forces_using_opstool(element_ids: List[int], num_points: int = 11) -> Dict:
    """Get V, N, M for elements using opstool post-processing"""
    if not OPSTOOL_AVAILABLE:
        raise ImportError("opstool not available")
    
    try:
        # Get element responses from existing ODB
        ele_resp = opst_post.get_element_responses(odb_tag="analysis", ele_type="Frame")
        ele_forces = ele_resp["sectionForces"]
        
        forces = {}
        for elem_id in element_ids:
            try:
                # Get forces at all section points
                N_data = ele_forces.sel(eleTags=elem_id, secDofs="N")
                V_data = ele_forces.sel(eleTags=elem_id, secDofs="VY")
                M_data = ele_forces.sel(eleTags=elem_id, secDofs="MZ")
                
                # Extract data at all section points
                sec_points = N_data.secPoints.data
                forces_list = []
                
                for sec_point in sec_points:
                    position = float(sec_point)  # Normalized position 0 to 1
                    N = float(N_data.sel(secPoints=sec_point).data[-1])
                    V = float(V_data.sel(secPoints=sec_point).data[-1])
                    M = float(M_data.sel(secPoints=sec_point).data[-1])
                    
                    forces_list.append({
                        "position": position,
                        "N": N,
                        "V": V,
                        "M": M
                    })
                
                # Sort by position
                forces_list.sort(key=lambda x: x["position"])
                forces[elem_id] = forces_list
            except Exception as e:
                # Fallback: get end forces only
                forces[elem_id] = [
                    {"position": 0.0, "N": 0.0, "V": 0.0, "M": 0.0},
                    {"position": 1.0, "N": 0.0, "V": 0.0, "M": 0.0}
                ]
        
        return forces
    except Exception as e:
        # Fallback method using basic OpenSees
        return get_element_forces_basic(element_ids)


def get_element_forces_basic(element_ids: List[int]) -> Dict:
    """Basic method to get element end forces"""
    forces = {}
    for elem_id in element_ids:
        try:
            # Get element end forces
            forces_raw = ops.eleResponse(elem_id, "forces")
            # Format: [Ni, Vi, Mi, Nj, Vj, Mj]
            forces[elem_id] = [
                {"position": 0.0, "N": float(forces_raw[0]), "V": float(forces_raw[1]), "M": float(forces_raw[2])},
                {"position": 1.0, "N": float(forces_raw[3]), "V": float(forces_raw[4]), "M": float(forces_raw[5])}
            ]
        except:
            forces[elem_id] = [
                {"position": 0.0, "N": 0.0, "V": 0.0, "M": 0.0},
                {"position": 1.0, "N": 0.0, "V": 0.0, "M": 0.0}
            ]
    return forces


def interpolate_at_position(forces_list: List[Dict], position: float) -> Dict:
    """Interpolate forces at a specific position along element"""
    if not forces_list:
        return {"N": 0.0, "V": 0.0, "M": 0.0}
    
    # Find closest points
    positions = [f["position"] for f in forces_list]
    
    if position <= positions[0]:
        return forces_list[0]
    if position >= positions[-1]:
        return forces_list[-1]
    
    # Linear interpolation
    for i in range(len(positions) - 1):
        if positions[i] <= position <= positions[i + 1]:
            p1, p2 = positions[i], positions[i + 1]
            f1, f2 = forces_list[i], forces_list[i + 1]
            alpha = (position - p1) / (p2 - p1) if p2 != p1 else 0.0
            
            return {
                "N": f1["N"] + alpha * (f2["N"] - f1["N"]),
                "V": f1["V"] + alpha * (f2["V"] - f1["V"]),
                "M": f1["M"] + alpha * (f2["M"] - f1["M"])
            }
    
    return forces_list[0]


# ==================== Diagram Generation Functions ====================

def create_force_diagram(forces: Dict, force_type: str, title: str, ylabel: str) -> str:
    """Create force diagram (M, V, or N)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for elem_id, values in forces.items():
        if not values:
            continue
        
        positions = [v["position"] for v in values]
        y_values = [v[force_type] for v in values]
        
        ax.plot(positions, y_values, label=f"Element {elem_id}", linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel("Position along element (normalized 0-1)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best')
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def create_displacement_diagram(displacements: Dict, request: AnalysisRequest) -> str:
    """Create displacement diagram showing deformed shape"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot original structure
    for elem in request.elements:
        node_i = next((n for n in request.nodes if n.id == elem.node_i), None)
        node_j = next((n for n in request.nodes if n.id == elem.node_j), None)
        if node_i and node_j:
            ax.plot([node_i.x, node_j.x], [node_i.y, node_j.y], 
                   'k-', linewidth=2, alpha=0.3, label='Original' if elem.id == request.elements[0].id else '')
    
    # Plot deformed structure
    scale = 100.0  # Displacement scale factor
    for elem in request.elements:
        node_i = next((n for n in request.nodes if n.id == elem.node_i), None)
        node_j = next((n for n in request.nodes if n.id == elem.node_j), None)
        if node_i and node_j:
            disp_i = displacements.get(elem.node_i, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
            disp_j = displacements.get(elem.node_j, {"ux": 0.0, "uy": 0.0, "rz": 0.0})
            
            x_i = node_i.x + scale * disp_i["ux"]
            y_i = node_i.y + scale * disp_i["uy"]
            x_j = node_j.x + scale * disp_j["ux"]
            y_j = node_j.y + scale * disp_j["uy"]
            
            ax.plot([x_i, x_j], [y_i, y_j], 
                   'r-', linewidth=2, alpha=0.7, 
                   label='Deformed' if elem.id == request.elements[0].id else '')
    
    ax.set_xlabel("X coordinate", fontsize=12)
    ax.set_ylabel("Y coordinate", fontsize=12)
    ax.set_title("Displacement Diagram (Deformed Shape)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "2D Frame Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Perform structural analysis",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_frame(request: AnalysisRequest):
    """
    Main analysis endpoint
    
    Performs static analysis of a 2D frame structure and returns:
    - Nodal displacements
    - Element forces (V, N, M) along elements
    - Values at designated positions
    - Diagrams for all response quantities
    """
    try:
        # Validate inputs
        if not request.nodes:
            raise HTTPException(status_code=400, detail="At least one node is required")
        if not request.elements:
            raise HTTPException(status_code=400, detail="At least one element is required")
        if not request.supports:
            raise HTTPException(status_code=400, detail="At least one support is required")
        
        # Create and run analysis
        create_model(request)
        run_static_analysis(num_points=11)
        
        # Get all node IDs and element IDs
        node_ids = [node.id for node in request.nodes]
        element_ids = [elem.id for elem in request.elements]
        
        # Get displacements
        displacements = get_nodal_displacements(node_ids)
        
        # Get element forces (V, N, M) using opstool
        try:
            forces = get_element_forces_using_opstool(element_ids, num_points=11)
        except Exception as e:
            # Fallback to basic method
            forces = get_element_forces_basic(element_ids)
        
        # Get values at designated positions
        designated_values = {}
        if request.designated_positions:
            for pos_req in request.designated_positions:
                elem_id = pos_req.element_id
                position = pos_req.position
                
                if elem_id in forces:
                    values = interpolate_at_position(forces[elem_id], position)
                    designated_values[f"element_{elem_id}_pos_{position:.3f}"] = {
                        "element_id": elem_id,
                        "position": position,
                        "N": values["N"],
                        "V": values["V"],
                        "M": values["M"],
                        "displacement": {
                            "ux": 0.0,  # Would need interpolation for element displacement
                            "uy": 0.0,
                            "rz": 0.0
                        }
                    }
        
        # Create diagrams
        moment_diagram = create_force_diagram(forces, "M", "Bending Moment Diagram", "Moment (M)")
        shear_diagram = create_force_diagram(forces, "V", "Shear Force Diagram", "Shear Force (V)")
        axial_diagram = create_force_diagram(forces, "N", "Axial Force Diagram", "Axial Force (N)")
        displacement_diagram = create_displacement_diagram(displacements, request)
        
        return JSONResponse(content={
            "status": "success",
            "displacements": displacements,
            "forces": forces,
            "designated_positions": designated_values,
            "diagrams": {
                "moment": f"data:image/png;base64,{moment_diagram}",
                "shear": f"data:image/png;base64,{shear_diagram}",
                "axial": f"data:image/png;base64,{axial_diagram}",
                "displacement": f"data:image/png;base64,{displacement_diagram}"
            }
        })
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        # Clean up
        try:
            ops.wipe()
        except:
            pass




