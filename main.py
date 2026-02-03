"""
Backend Server for Two-Dimensional Moment Frame Analysis
=========================================================
FastAPI server providing REST API endpoints for frame analysis operations.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import openseespy.opensees as ops
import opstool as opst
import json
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Frame Analysis API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class LoadPattern(BaseModel):
    node_id: int
    fx: float
    fy: float = 0.0
    mz: float = 0.0


class StaticAnalysisRequest(BaseModel):
    loads: List[LoadPattern]
    n_steps: int = 10


class SeismicAnalysisRequest(BaseModel):
    time: List[float]
    acceleration: List[float]
    damping_ratio: float = 0.05
    n_steps: int = 1600
    dt: float = 0.02


class EigenAnalysisRequest(BaseModel):
    mode_tag: int = 6


def FEModel():
    """Create the finite element model"""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    # Defining Nodes
    ops.node(1, 0.000000e00, 0.000000e00)
    ops.node(2, 3.600000e02, 0.000000e00)
    ops.node(3, 7.200000e02, 0.000000e00)
    ops.node(4, 0.000000e00, 1.620000e02)
    ops.node(5, 3.600000e02, 1.620000e02)
    ops.node(6, 7.200000e02, 1.620000e02)
    ops.node(7, 0.000000e00, 3.240000e02)
    ops.node(8, 3.600000e02, 3.240000e02)
    ops.node(9, 7.200000e02, 3.240000e02)
    ops.node(10, 0.000000e00, 4.800000e02)
    ops.node(11, 3.600000e02, 4.800000e02)
    ops.node(12, 7.200000e02, 4.800000e02)
    ops.node(13, 0.000000e00, 6.360000e02)
    ops.node(14, 3.600000e02, 6.360000e02)
    ops.node(15, 7.200000e02, 6.360000e02)
    ops.node(16, 0.000000e00, 7.920000e02)
    ops.node(17, 3.600000e02, 7.920000e02)
    ops.node(18, 7.200000e02, 7.920000e02)
    ops.node(19, 0.000000e00, 9.480000e02)
    ops.node(20, 3.600000e02, 9.480000e02)
    ops.node(21, 7.200000e02, 9.480000e02)
    ops.node(22, 0.000000e00, 1.104000e03)
    ops.node(23, 3.600000e02, 1.104000e03)
    ops.node(24, 7.200000e02, 1.104000e03)

    # Write node restraint
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)
    ops.fix(3, 1, 1, 1)

    # Define the rigidDiaphragm
    ops.rigidDiaphragm(1, 5, 4, 6)
    ops.rigidDiaphragm(1, 8, 7, 9)
    ops.rigidDiaphragm(1, 11, 10, 12)
    ops.rigidDiaphragm(1, 14, 13, 15)
    ops.rigidDiaphragm(1, 17, 16, 18)
    ops.rigidDiaphragm(1, 20, 19, 21)
    ops.rigidDiaphragm(1, 23, 22, 24)

    # Defining Frame Elements
    ops.geomTransf("Linear", 1)
    ops.element("elasticBeamColumn", 1, 1, 4, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 2, 4, 7, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 3, 7, 10, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 4, 10, 13, 6.210000e01, 2.950000e04, 2.670000e03, 1)
    ops.element("elasticBeamColumn", 5, 13, 16, 6.210000e01, 2.950000e04, 2.670000e03, 1)
    ops.element("elasticBeamColumn", 6, 16, 19, 5.170000e01, 2.950000e04, 2.150000e03, 1)
    ops.element("elasticBeamColumn", 7, 19, 22, 5.170000e01, 2.950000e04, 2.150000e03, 1)
    ops.element("elasticBeamColumn", 8, 2, 5, 8.440000e01, 2.950000e04, 3.910000e03, 1)
    ops.element("elasticBeamColumn", 9, 5, 8, 8.440000e01, 2.950000e04, 3.910000e03, 1)
    ops.element("elasticBeamColumn", 10, 8, 11, 8.440000e01, 2.950000e04, 3.910000e03, 1)
    ops.element("elasticBeamColumn", 11, 11, 14, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 12, 14, 17, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 13, 17, 20, 6.210000e01, 2.950000e04, 2.670000e03, 1)
    ops.element("elasticBeamColumn", 14, 20, 23, 6.210000e01, 2.950000e04, 2.670000e03, 1)
    ops.element("elasticBeamColumn", 15, 3, 6, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 16, 6, 9, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 17, 9, 12, 7.230000e01, 2.950000e04, 3.230000e03, 1)
    ops.element("elasticBeamColumn", 18, 12, 15, 6.210000e01, 2.950000e04, 2.670000e03, 1)
    ops.element("elasticBeamColumn", 19, 15, 18, 6.210000e01, 2.950000e04, 2.670000e03, 1)
    ops.element("elasticBeamColumn", 20, 18, 21, 5.170000e01, 2.950000e04, 2.150000e03, 1)
    ops.element("elasticBeamColumn", 21, 21, 24, 5.170000e01, 2.950000e04, 2.150000e03, 1)
    ops.element("elasticBeamColumn", 22, 4, 5, 4.710000e01, 2.950000e04, 5.120000e03, 1)
    ops.element("elasticBeamColumn", 23, 7, 8, 4.710000e01, 2.950000e04, 5.120000e03, 1)
    ops.element("elasticBeamColumn", 24, 10, 11, 3.830000e01, 2.950000e04, 4.020000e03, 1)
    ops.element("elasticBeamColumn", 25, 13, 14, 3.830000e01, 2.950000e04, 4.020000e03, 1)
    ops.element("elasticBeamColumn", 26, 16, 17, 3.250000e01, 2.950000e04, 3.330000e03, 1)
    ops.element("elasticBeamColumn", 27, 19, 20, 3.250000e01, 2.950000e04, 3.330000e03, 1)
    ops.element("elasticBeamColumn", 28, 22, 23, 3.250000e01, 2.950000e04, 3.330000e03, 1)
    ops.element("elasticBeamColumn", 29, 5, 6, 4.710000e01, 2.950000e04, 5.120000e03, 1)
    ops.element("elasticBeamColumn", 30, 8, 9, 4.710000e01, 2.950000e04, 5.120000e03, 1)
    ops.element("elasticBeamColumn", 31, 11, 12, 3.830000e01, 2.950000e04, 4.020000e03, 1)
    ops.element("elasticBeamColumn", 32, 14, 15, 3.830000e01, 2.950000e04, 4.020000e03, 1)
    ops.element("elasticBeamColumn", 33, 17, 18, 3.250000e01, 2.950000e04, 3.330000e03, 1)
    ops.element("elasticBeamColumn", 34, 20, 21, 3.250000e01, 2.950000e04, 3.330000e03, 1)
    ops.element("elasticBeamColumn", 35, 23, 24, 3.250000e01, 2.950000e04, 3.330000e03, 1)

    # Define the mass
    ops.mass(5, 0.49, 0.0, 0.0)
    ops.mass(8, 0.49, 0.0, 0.0)
    ops.mass(11, 0.49, 0.0, 0.0)
    ops.mass(14, 0.49, 0.0, 0.0)
    ops.mass(17, 0.49, 0.0, 0.0)
    ops.mass(20, 0.49, 0.0, 0.0)
    ops.mass(23, 0.49, 0.0, 0.0)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Frame Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "eigen": "/api/eigen",
            "static": "/api/static",
            "seismic": "/api/seismic",
            "docs": "/docs"
        }
    }


@app.post("/api/eigen")
async def eigenvalue_analysis(request: EigenAnalysisRequest):
    """Perform eigenvalue analysis"""
    try:
        FEModel()
        opst.post.save_eigen_data(odb_tag="eigen", mode_tag=request.mode_tag, solver="-fullGenLapack")
        
        model_props, eigen_vectors = opst.post.get_eigen_data(odb_tag="eigen")
        model_props_df = model_props.to_pandas()
        
        return {
            "status": "success",
            "modal_periods": model_props_df["eigenPeriod"].tolist(),
            "participation_mass_ratios": model_props_df["partiMassRatiosMX"].tolist(),
            "cumulative_participation_mass_ratios": model_props_df["partiMassRatiosCumuMX"].tolist(),
            "data": model_props_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/static")
async def static_analysis(request: StaticAnalysisRequest):
    """Perform static analysis"""
    try:
        FEModel()
        
        # Define the load pattern
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        
        for load in request.loads:
            ops.load(load.node_id, load.fx, load.fy, load.mz)
        
        # Perform analysis
        n_steps = request.n_steps
        ops.system("BandGeneral")
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.test("NormDispIncr", 1.0e-12, 10, 3)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 1 / n_steps)
        ops.analysis("Static")
        
        # Save the results
        odb = opst.post.CreateODB(odb_tag="static", interpolate_beam_disp=11)
        for _ in range(n_steps):
            ops.analyze(1)
            odb.fetch_response_step()
        odb.save_response()
        
        # Retrieve nodal responses
        node_resp = opst.post.get_nodal_responses(odb_tag="static")
        
        # Retrieve element responses
        ele_resp = opst.post.get_element_responses(odb_tag="static", ele_type="Frame")
        
        # Convert to JSON-serializable format
        result = {
            "status": "success",
            "nodal_displacements": {},
            "element_forces": {}
        }
        
        # Extract nodal displacements for all nodes
        for node_tag in range(1, 25):
            try:
                disp_x = float(node_resp["disp"].sel(nodeTags=node_tag, DOFs="UX").data[-1])
                disp_y = float(node_resp["disp"].sel(nodeTags=node_tag, DOFs="UY").data[-1])
                result["nodal_displacements"][str(node_tag)] = {
                    "UX": disp_x,
                    "UY": disp_y
                }
            except:
                pass
        
        # Extract element forces for a few key elements
        ele_forces = ele_resp["sectionForces"]
        for ele_tag in [1, 8, 15, 22]:
            try:
                M = float(ele_forces.sel(eleTags=ele_tag, secDofs="MZ", secPoints=1).data[-1])
                V = float(ele_forces.sel(eleTags=ele_tag, secDofs="VY", secPoints=1).data[-1])
                N = float(ele_forces.sel(eleTags=ele_tag, secDofs="N", secPoints=1).data[-1])
                result["element_forces"][str(ele_tag)] = {
                    "M": M,
                    "V": V,
                    "N": N
                }
            except:
                pass
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/seismic")
async def seismic_analysis(request: SeismicAnalysisRequest):
    """Perform seismic response analysis"""
    try:
        FEModel()
        
        # Create the ground motion load pattern
        time = np.array(request.time)
        accel = np.array(request.acceleration)
        ops.timeSeries("Path", 2, "-time", *time, "-values", *accel, "-factor", 386.4)
        ops.pattern("UniformExcitation", 2, 1, "-accel", 2)
        
        # Create the Rayleigh damping
        xDamp = request.damping_ratio
        MpropSwitch = 1.0
        KcurrSwitch = 0.0
        KcommSwitch = 1.0
        KinitSwitch = 0.0
        nEigenI = 1
        nEigenJ = 2
        lambdaN = ops.eigen(nEigenJ)
        lambdaI = lambdaN[nEigenI - 1]
        lambdaJ = lambdaN[nEigenJ - 1]
        omegaI = np.sqrt(lambdaI)
        omegaJ = np.sqrt(lambdaJ)
        alphaM = MpropSwitch * xDamp * (2 * omegaI * omegaJ) / (omegaI + omegaJ)
        betaKcurr = KcurrSwitch * 2.0 * xDamp / (omegaI + omegaJ)
        betaKcomm = KcommSwitch * 2.0 * xDamp / (omegaI + omegaJ)
        betaKinit = KinitSwitch * 2.0 * xDamp / (omegaI + omegaJ)
        ops.rayleigh(alphaM, 0.0, 0.0, betaKcomm)
        
        # Perform analysis
        ops.wipeAnalysis()
        ops.system("BandGeneral")
        ops.constraints("Transformation")
        ops.numberer("RCM")
        ops.test("NormDispIncr", 1.0e-12, 10, 3)
        ops.algorithm("Linear")
        ops.integrator("HHT", 1.0, 0.5, 0.25)
        ops.analysis("Transient")
        
        n_steps = request.n_steps
        dt = request.dt
        odb = opst.post.CreateODB(odb_tag="seismic", interpolate_beam_disp=11)
        for _ in range(n_steps):
            ops.analyze(1, dt)
            odb.fetch_response_step()
        odb.save_response()
        
        # Retrieve node responses
        node_resp = opst.post.get_nodal_responses(odb_tag="seismic")
        node_disp22 = node_resp["disp"].sel(nodeTags=22, DOFs="UX")
        node_vel22 = node_resp["vel"].sel(nodeTags=22, DOFs="UX")
        node_accel22 = node_resp["accel"].sel(nodeTags=22, DOFs="UX")
        
        # Retrieve element responses
        ele_resp = opst.post.get_element_responses(odb_tag="seismic", ele_type="Frame", resp_type="sectionForces")
        frame1Mz = -ele_resp.sel(eleTags=1, secDofs="MZ", secPoints=1)
        frame1N = ele_resp.sel(eleTags=1, secDofs="N", secPoints=1)
        frame1Vy = ele_resp.sel(eleTags=1, secDofs="VY", secPoints=1)
        
        return {
            "status": "success",
            "node_22": {
                "displacement": {
                    "max": float(node_disp22.data.max()),
                    "min": float(node_disp22.data.min()),
                    "time_series": node_disp22.time.tolist(),
                    "values": node_disp22.data.tolist()
                },
                "velocity": {
                    "max": float(node_vel22.data.max()),
                    "min": float(node_vel22.data.min()),
                    "values": node_vel22.data.tolist()
                },
                "acceleration": {
                    "max": float(node_accel22.data.max()),
                    "min": float(node_accel22.data.min()),
                    "values": node_accel22.data.tolist()
                }
            },
            "element_1": {
                "moment": {
                    "max": float(frame1Mz.data.max()),
                    "min": float(frame1Mz.data.min()),
                    "values": frame1Mz.data.tolist()
                },
                "shear": {
                    "max": float(frame1Vy.data.max()),
                    "min": float(frame1Vy.data.min()),
                    "values": frame1Vy.data.tolist()
                },
                "axial": {
                    "max": float(frame1N.data.max()),
                    "min": float(frame1N.data.min()),
                    "values": frame1N.data.tolist()
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/seismic/upload")
async def seismic_analysis_upload(file: UploadFile = File(...), damping_ratio: float = 0.05, n_steps: int = 1600, dt: float = 0.02):
    """Perform seismic analysis with uploaded ground motion file"""
    try:
        contents = await file.read()
        gmdata = np.loadtxt(io.BytesIO(contents))
        time = gmdata[:, 0].tolist()
        accel = gmdata[:, 1].tolist()
        
        request = SeismicAnalysisRequest(
            time=time,
            acceleration=accel,
            damping_ratio=damping_ratio,
            n_steps=n_steps,
            dt=dt
        )
        
        return await seismic_analysis(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host=host, port=port)


