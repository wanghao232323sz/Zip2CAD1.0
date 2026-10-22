"""Standalone SketchUp C API bindings. No Ruby, UI automation or CAD dimensions.

Provide your own SketchUpAPI library through ZIP2CAD_SKETCHUP_SDK or --sdk.
The proprietary library is never copied into this project or release archive.
All SDK calls execute on the calling process's main thread, per SDK contract.
"""
from pathlib import Path
import ctypes as ct
import hashlib,json,os,threading
import numpy as np

class SDKError(RuntimeError):pass
class Ref(ct.Structure):_fields_=[('ptr',ct.c_void_p)]
class Point(ct.Structure):_fields_=[('x',ct.c_double),('y',ct.c_double),('z',ct.c_double)]
class Point2(ct.Structure):_fields_=[('x',ct.c_double),('y',ct.c_double)]
class Transform(ct.Structure):_fields_=[('values',ct.c_double*16)]
class Color(ct.Structure):_fields_=[('red',ct.c_ubyte),('green',ct.c_ubyte),('blue',ct.c_ubyte),('alpha',ct.c_ubyte)]
class MaterialInput(ct.Structure):_fields_=[('num_uv_coords',ct.c_size_t),('uv_coords',Point2*4),('vertex_indices',ct.c_size_t*4),('material',Ref)]
PRef=ct.POINTER(Ref);Size=ct.c_size_t;PSize=ct.POINTER(Size)

def discover_sdk(path=None):
    requested=path or os.environ.get('ZIP2CAD_SKETCHUP_SDK')
    if requested:
        p=Path(requested).expanduser().resolve()
        if p.is_dir():
            choices=[p/'SketchUpAPI.dll',p/'binaries/sketchup/x64/SketchUpAPI.dll',p/'SketchUpAPI.framework/SketchUpAPI']
            p=next((f for f in choices if f.is_file()),p)
        if not p.is_file():raise SDKError('SDK library not found: '+str(p))
        return p
    root=Path(os.environ.get('ProgramFiles','C:/Program Files'))/'SketchUp'
    choices=sorted(root.glob('SketchUp */SketchUp/SketchUpAPI.dll'),reverse=True)
    if choices:return choices[0]
    raise SDKError('Provide --sdk /path/to/SketchUpAPI.dll or set ZIP2CAD_SKETCHUP_SDK. See docs/HEADLESS_TOOL.md.')

class SketchUpSDK:
    def __init__(self,path=None):
        if threading.current_thread() is not threading.main_thread():raise SDKError('SketchUp C API requires the main thread')
        self.library_path=discover_sdk(path);self._directories=[];self._closed=False
        if os.name=='nt':self._directories.append(os.add_dll_directory(str(self.library_path.parent)))
        self.dll=ct.CDLL(str(self.library_path));self.functions={};self.mesh_cache={}
        self.bind('SUInitialize',[],None);self.bind('SUTerminate',[],None)
        self.bind('SUModelCreateFromFileWithStatus',[PRef,ct.c_char_p,ct.POINTER(ct.c_int)])
        self.bind('SUModelCreate',[PRef]);self.bind('SUModelRelease',[PRef]);self.bind('SUModelSaveToFile',[Ref,ct.c_char_p])
        self.bind('SUModelGetEntities',[Ref,PRef]);self.bind('SUEntityGetPersistentID',[Ref,ct.POINTER(ct.c_int64)])
        self.bind('SUStringCreate',[PRef]);self.bind('SUStringRelease',[PRef]);self.bind('SUStringGetUTF8Length',[Ref,PSize]);self.bind('SUStringGetUTF8',[Ref,Size,ct.c_void_p,PSize])
        for kind in ['Faces','Edges','Groups','Instances','Images','GuidePoints','GuideLines','Texts','SectionPlanes']:
            extra=[ct.c_bool] if kind=='Edges' else []
            self.bind('SUEntitiesGetNum'+kind,[Ref,*extra,PSize]);self.bind('SUEntitiesGet'+kind,[Ref,*extra,Size,PRef,PSize])
        for kind in ['Face','Edge','Group','ComponentInstance','Image','GuidePoint','GuideLine','Text','SectionPlane','Material','Layer']:
            self.bind('SU'+kind+'ToEntity',[Ref],Ref)
        self.bind('SUDrawingElementFromEntity',[Ref],Ref)
        self.bind('SUDrawingElementGetHidden',[Ref,ct.POINTER(ct.c_bool)]);self.bind('SUDrawingElementSetHidden',[Ref,ct.c_bool])
        self.bind('SUDrawingElementGetLayer',[Ref,PRef]);self.bind('SUDrawingElementGetMaterial',[Ref,PRef])
        self.bind('SULayerGetVisibility',[Ref,ct.POINTER(ct.c_bool)])
        for name in ['SULayerGetName','SUMaterialGetName','SUGroupGetName','SUComponentInstanceGetName']:self.bind(name,[Ref,PRef])
        for name in ['SUGroupGetTransform','SUComponentInstanceGetTransform']:self.bind(name,[Ref,ct.POINTER(Transform)])
        for name in ['SUGroupGetEntities','SUComponentInstanceGetDefinition','SUComponentDefinitionGetEntities']:self.bind(name,[Ref,PRef])
        self.bind('SUMeshHelperCreate',[PRef,Ref]);self.bind('SUMeshHelperRelease',[PRef])
        for name in ['SUMeshHelperGetNumVertices','SUMeshHelperGetNumTriangles']:self.bind(name,[Ref,PSize])
        self.bind('SUMeshHelperGetVertices',[Ref,Size,ct.POINTER(Point),PSize]);self.bind('SUMeshHelperGetVertexIndices',[Ref,Size,PSize,PSize])
        for name in ['SUMeshHelperGetFrontSTQCoords','SUMeshHelperGetBackSTQCoords']:self.bind(name,[Ref,Size,ct.POINTER(Point),PSize])
        self.bind('SUEdgeGetStartVertex',[Ref,PRef]);self.bind('SUEdgeGetEndVertex',[Ref,PRef]);self.bind('SUVertexGetPosition',[Ref,ct.POINTER(Point)])
        self.bind('SUEdgeGetNumFaces',[Ref,PSize]);self.bind('SUEdgeSetSoft',[Ref,ct.c_bool]);self.bind('SUEdgeSetSmooth',[Ref,ct.c_bool])
        for name in ['SUEdgeGetSoft','SUEdgeGetSmooth']:self.bind(name,[Ref,ct.POINTER(ct.c_bool)])
        for name in ['SUFaceGetFrontMaterial','SUFaceGetBackMaterial']:self.bind(name,[Ref,PRef])
        for name in ['SUFaceSetFrontMaterial','SUFaceSetBackMaterial']:self.bind(name,[Ref,Ref])
        self.bind('SUComponentDefinitionIsManifold',[Ref,ct.POINTER(ct.c_bool)])
        self.bind('SUGroupGetDefinition',[Ref,PRef])
        self.bind('SUGroupCreate',[PRef]);self.bind('SUGroupSetName',[Ref,ct.c_char_p]);self.bind('SUEntitiesAddGroup',[Ref,Ref])
        self.bind('SUGeometryInputCreate',[PRef]);self.bind('SUGeometryInputRelease',[PRef]);self.bind('SUGeometryInputSetVertices',[Ref,Size,ct.POINTER(Point)])
        self.bind('SULoopInputCreate',[PRef]);self.bind('SULoopInputRelease',[PRef]);self.bind('SULoopInputAddVertexIndex',[Ref,Size])
        self.bind('SUGeometryInputAddFace',[Ref,PRef,PSize]);self.bind('SUEntitiesFill',[Ref,Ref,ct.c_bool])
        self.bind('SUMaterialCreate',[PRef]);self.bind('SUMaterialSetName',[Ref,ct.c_char_p]);self.bind('SUMaterialSetColor',[Ref,ct.POINTER(Color)])
        self.bind('SUMaterialGetColor',[Ref,ct.POINTER(Color)]);self.bind('SUMaterialGetOpacity',[Ref,ct.POINTER(ct.c_double)])
        self.bind('SUMaterialGetTexture',[Ref,PRef]);self.bind('SUTextureGetImageRep',[Ref,PRef]);self.bind('SUImageRepCreate',[PRef]);self.bind('SUImageRepRelease',[PRef])
        self.bind('SUImageRepGetDataSize',[Ref,PSize,PSize]);self.bind('SUImageRepGetData',[Ref,Size,ct.c_void_p])
        self.bind('SUModelGetNumMaterials',[Ref,PSize]);self.bind('SUModelGetMaterials',[Ref,Size,PRef,PSize]);self.bind('SUModelAddMaterials',[Ref,Size,PRef])
        self.bind('SUGeometryInputFaceSetFrontMaterial',[Ref,Size,ct.POINTER(MaterialInput)])
        self.bind('SUGeometryInputFaceSetBackMaterial',[Ref,Size,ct.POINTER(MaterialInput)])
        self.bind('SUCameraCreate',[PRef]);self.bind('SUCameraSetOrientation',[Ref,ct.POINTER(Point),ct.POINTER(Point),ct.POINTER(Point)])
        self.bind('SUCameraSetPerspective',[Ref,ct.c_bool]);self.bind('SUCameraSetPerspectiveFrustumFOV',[Ref,ct.c_double]);self.bind('SUModelSetCamera',[Ref,PRef]);self.bind('SUModelGetCamera',[Ref,PRef])
        self.bind('SUSceneCreate',[PRef]);self.bind('SUSceneSetName',[Ref,ct.c_char_p]);self.bind('SUSceneSetCamera',[Ref,Ref])
        for name in ['SUSceneSetUseCamera','SUSceneSetUseHiddenGeometry','SUSceneSetUseHiddenObjects']:self.bind(name,[Ref,ct.c_bool])
        self.bind('SUModelAddScenes',[Ref,Size,PRef]);self.bind('SUModelSetActiveScene',[Ref,Ref])
        self.bind('SUModelGetNumScenes',[Ref,PSize]);self.bind('SUModelGetScenes',[Ref,Size,PRef,PSize]);self.bind('SUSceneGetName',[Ref,PRef])
        self.bind('SUEntitiesTransform',[Ref,Size,PRef,ct.POINTER(Transform)])
        self.dll.SUInitialize()
    def bind(self,name,args,result=ct.c_int):
        fn=getattr(self.dll,name);fn.argtypes=args;fn.restype=result;self.functions[name]=fn
    def call(self,name,*args,allow=()):
        code=self.functions[name](*args)
        if code!=0 and code not in allow:raise SDKError(f'{name} returned SUResult {code}')
        return code
    def ref(self,name,*args):
        result=Ref();self.call(name,*args,ct.byref(result));return result
    def number(self,name,*args):
        value=Size();self.call(name,*args,ct.byref(value));return value.value
    def text(self,name,obj):
        s=self.ref('SUStringCreate')
        try:
            self.call(name,obj,ct.byref(s));n=self.number('SUStringGetUTF8Length',s);buf=ct.create_string_buffer(n+1);count=Size()
            self.call('SUStringGetUTF8',s,n+1,buf,ct.byref(count));return buf.value.decode('utf-8')
        finally:self.call('SUStringRelease',ct.byref(s))
    def entity(self,kind,obj):return self.functions['SU'+kind+'ToEntity'](obj)
    def pid(self,kind,obj):
        n=ct.c_int64();self.call('SUEntityGetPersistentID',self.entity(kind,obj),ct.byref(n));return n.value
    def list(self,entities,kind):
        extra=[False] if kind=='Edges' else []
        n=self.number('SUEntitiesGetNum'+kind,entities,*extra)
        if not n:return []
        array=(Ref*n)();count=Size();self.call('SUEntitiesGet'+kind,entities,*extra,n,array,ct.byref(count));return list(array)[:count.value]
    def load(self,path):
        model=Ref();status=ct.c_int();self.call('SUModelCreateFromFileWithStatus',ct.byref(model),str(Path(path).resolve()).encode('utf-8'),ct.byref(status))
        if status.value!=0:
            self.release(model);raise SDKError('Model was saved by a newer SDK; preserving original requires a compatible SDK')
        return model
    def release(self,model):
        self.call('SUModelRelease',ct.byref(model));self.mesh_cache.clear()
    def __enter__(self):return self
    def __exit__(self,*args):
        if not self._closed:self.dll.SUTerminate();self._closed=True
        for directory in self._directories:directory.close()
    def transform(self,kind,obj):
        value=Transform();self.call('SU'+kind+'GetTransform',obj,ct.byref(value));matrix=np.array(value.values).reshape(4,4,order='F');matrix[:3,3]*=.0254;return matrix
    def children(self,kind,obj):
        if kind=='Group':return self.ref('SUGroupGetEntities',obj)
        definition=self.ref('SUComponentInstanceGetDefinition',obj);return self.ref('SUComponentDefinitionGetEntities',definition)
    def drawing_record(self,kind,obj):
        d=self.functions['SUDrawingElementFromEntity'](self.entity(kind,obj));hidden=ct.c_bool();self.call('SUDrawingElementGetHidden',d,ct.byref(hidden))
        layer=self.ref('SUDrawingElementGetLayer',d);material=Ref();code=self.call('SUDrawingElementGetMaterial',d,ct.byref(material),allow=(9,))
        visible=ct.c_bool();self.call('SULayerGetVisibility',layer,ct.byref(visible))
        return {'hidden':hidden.value,'layer':self.text('SULayerGetName',layer),'layer_visible':visible.value,'material':self.pid('Material',material) if code==0 and material.ptr else None}
    def hidden(self,kind,obj,value):self.call('SUDrawingElementSetHidden',self.functions['SUDrawingElementFromEntity'](self.entity(kind,obj)),bool(value))
    def face_mesh(self,face):
        key=face.ptr
        if key in self.mesh_cache:return self.mesh_cache[key]
        mesh=Ref();self.call('SUMeshHelperCreate',ct.byref(mesh),face)
        try:
            nv=self.number('SUMeshHelperGetNumVertices',mesh);nt=self.number('SUMeshHelperGetNumTriangles',mesh);v=(Point*nv)();f=(Size*(nt*3))();count=Size()
            self.call('SUMeshHelperGetVertices',mesh,nv,v,ct.byref(count));self.call('SUMeshHelperGetVertexIndices',mesh,nt*3,f,ct.byref(count))
            points=np.array([[p.x,p.y,p.z] for p in v])*.0254;faces=np.array(f,dtype=np.int64).reshape(-1,3)
            uv=[]
            for side in ['Front','Back']:
                data=(Point*nv)();code=self.call('SUMeshHelperGet'+side+'STQCoords',mesh,nv,data,ct.byref(count),allow=(9,))
                # Texture STQ is dimensionless and can exceed 50,000. SDK
                # serialization perturbs its last bits; geometry stays at 1e-8 m.
                uv.append(np.array([[p.x,p.y,p.z] for p in data]).round(6).tolist() if code==0 else None)
            result=(points,faces,uv);self.mesh_cache[key]=result;return result
        finally:self.call('SUMeshHelperRelease',ct.byref(mesh))
    def walk(self,model,scope=None):
        root=self.ref('SUModelGetEntities',model)
        def descend(entities,path,matrix,ancestry):
            if entities.ptr in ancestry or len(path)>100:raise SDKError('Cyclic or excessively nested model')
            ancestry=ancestry|{entities.ptr}
            mapping={'Faces':'Face','Edges':'Edge','Groups':'Group','Instances':'ComponentInstance','Images':'Image','GuidePoints':'GuidePoint','GuideLines':'GuideLine','Texts':'Text','SectionPlanes':'SectionPlane'}
            for plural,kind in mapping.items():
                if scope and len(path)<len(scope)-1 and kind not in ('Group','ComponentInstance'):continue
                for obj in sorted(self.list(entities,plural),key=lambda obj:self.pid(kind,obj)):
                    ids=path+[self.pid(kind,obj)]
                    if scope and ids[:min(len(ids),len(scope))]!=list(scope)[:min(len(ids),len(scope))]:continue
                    local=self.transform(kind,obj) if kind in ('Group','ComponentInstance') else np.eye(4)
                    world=matrix@local;yield kind,obj,ids,world,local
                    if kind in ('Group','ComponentInstance'):yield from descend(self.children(kind,obj),ids,world,ancestry)
        yield from descend(root,[],np.eye(4),set())
    def geometry(self,model,namespace='building',scope=None,exclude=None,include_appearance=False,visible_only=False):
        vertices=[];faces=[];owners=[];ids=[];rows=[];offset=0;opacities=[];colors=[];parent_opacity={};parent_hidden={};parent_color={}
        materials={r['pid']:r for r in self.materials(model)} if include_appearance else {}
        for kind,obj,path,world,local in self.walk(model,scope):
            if include_appearance or visible_only:
                drawing=self.drawing_record(kind,obj);material=drawing['material'];opacity=materials[material]['opacity'] if material in materials else parent_opacity.get(tuple(path[:-1]),1.)
                color=materials[material]['rgba'][:3] if material in materials else parent_color.get(tuple(path[:-1]),[240,240,240])
                hidden=drawing['hidden'] or not drawing['layer_visible'] or parent_hidden.get(tuple(path[:-1]),False)
                if kind in ('Group','ComponentInstance'):
                    parent_opacity[tuple(path)]=opacity;parent_color[tuple(path)]=color;parent_hidden[tuple(path)]=hidden
                if visible_only and hidden:continue
            if exclude and path[:len(exclude)]==list(exclude):continue
            if scope and path[:len(scope)]!=list(scope):continue
            if kind!='Face':continue
            p,f,_=self.face_mesh(obj);p=p@world[:3,:3].T+world[:3,3]
            if np.linalg.det(world[:3,:3])<0:f=f[:,::-1]
            vertices.append(p);faces.append(f+offset);owners.extend([len(ids)]*len(f));offset+=len(p);ids.append('skp:'+namespace+':'+'.'.join(map(str,path)))
            opacities.append(opacity if include_appearance else 1.)
            colors.append(color if include_appearance else [240,240,240])
            rows.append({'pid_path':path,'triangle_count':len(f),'bounds_m':[p.min(0).tolist(),p.max(0).tolist()]})
        return {'vertices_world_m':np.vstack(vertices) if vertices else np.empty((0,3)), 'faces':np.vstack(faces) if faces else np.empty((0,3),int),
            'face_owner_indices':np.array(owners,int),'entity_ids':np.array(ids),'face_material_opacity':np.array(opacities),'face_material_rgb':np.array(colors),
            'source_face_refs':np.array([p['pid_path'][-1] for p in rows]),'rows':rows}
    def snapshot(self,model,exclude=None,hidden_overrides=None,appearance_overrides=None):
        hidden_overrides=hidden_overrides or {};digest=hashlib.sha256();counts={};cache={};self.last_snapshot_hashes={};self.debug_records={}
        for kind,obj,path,world,local in self.walk(model):
            if exclude and path[:len(exclude)]==list(exclude):continue
            counts[kind]=counts.get(kind,0)+1;record={'path':path,'kind':kind,'drawing':self.drawing_record(kind,obj)}
            key='.'.join(map(str,path));record['drawing']['hidden']=hidden_overrides.get(key,record['drawing']['hidden'])
            if kind in ('Group','ComponentInstance'):record.update(transform=local.round(8).tolist(),name=self.text('SU'+kind+'GetName',obj))
            elif kind=='Face':
                if obj.ptr not in cache:
                    p,f,uv=self.face_mesh(obj);cache[obj.ptr]=[p.round(8).tolist(),f.tolist(),uv]
                record['mesh']=cache[obj.ptr]
                record['face_materials']=[]
                for side in ['Front','Back']:
                    material=Ref();code=self.call('SUFaceGet'+side+'Material',obj,ct.byref(material),allow=(9,))
                    record['face_materials'].append(self.pid('Material',material) if code==0 and material.ptr else None)
            elif kind=='Edge':
                points=[]
                for end in ['Start','End']:
                    v=self.ref('SUEdgeGet'+end+'Vertex',obj);p=Point();self.call('SUVertexGetPosition',v,ct.byref(p));points.append([round(p.x*.0254,8),round(p.y*.0254,8),round(p.z*.0254,8)])
                record['points']=points
                for name in ['Soft','Smooth']:
                    value=ct.c_bool();self.call('SUEdgeGet'+name,obj,ct.byref(value));record[name]=value.value
            if kind=='Face' and appearance_overrides and key in appearance_overrides:
                original=appearance_overrides[key];record['drawing']['material']=original['drawing']['material'];record['face_materials']=original['face_materials']
                # UV association is part of the authorized finish edit; vertex
                # positions, topology, hierarchy and all other faces stay strict.
                record['mesh']=[record['mesh'][0],record['mesh'][1],original['mesh'][2]]
            data=json.dumps(record,sort_keys=True,separators=(',',':')).encode();self.last_snapshot_hashes[key]=hashlib.sha256(data).hexdigest()
            if key in getattr(self,'debug_paths',set()):self.debug_records[key]=record
            digest.update(data);digest.update(b'\n')
        return {'sha256':digest.hexdigest(),'counts':counts}
    def materials(self,model):
        n=self.number('SUModelGetNumMaterials',model);array=(Ref*n)();count=Size();self.call('SUModelGetMaterials',model,n,array,ct.byref(count));rows=[]
        for m in list(array)[:count.value]:
            color=Color();self.call('SUMaterialGetColor',m,ct.byref(color),allow=(9,));opacity=ct.c_double();self.call('SUMaterialGetOpacity',m,ct.byref(opacity))
            texture=Ref();code=self.call('SUMaterialGetTexture',m,ct.byref(texture),allow=(9,));texture_hash=None
            if code==0 and texture.ptr:
                rep=self.ref('SUImageRepCreate')
                try:
                    self.call('SUTextureGetImageRep',texture,ct.byref(rep));size=Size();bits=Size();self.call('SUImageRepGetDataSize',rep,ct.byref(size),ct.byref(bits));data=ct.create_string_buffer(size.value);self.call('SUImageRepGetData',rep,size.value,data);texture_hash=hashlib.sha256(data.raw).hexdigest()
                finally:self.call('SUImageRepRelease',ct.byref(rep))
            rows.append({'pid':self.pid('Material',m),'name':self.text('SUMaterialGetName',m),'rgba':[color.red,color.green,color.blue,color.alpha],'opacity':opacity.value,'texture_sha256':texture_hash})
        return sorted(rows,key=lambda r:r['pid'])
    def group(self,entities,name):
        group=self.ref('SUGroupCreate');self.call('SUGroupSetName',group,name.encode('utf-8'));self.call('SUEntitiesAddGroup',entities,group);return group
    def add_solid(self,model,parent,item):
        group=self.group(parent,'Zip2CAD '+item['component_id']);entities=self.children('Group',group)
        materials=[]
        for side in ['top','side']:
            m=self.ref('SUMaterialCreate');self.call('SUMaterialSetName',m,('Zip2CAD_'+item['component_id']+'_'+side).encode());rgba=Color(*item[side+'_color'],255);self.call('SUMaterialSetColor',m,ct.byref(rgba));materials.append(m)
        self.call('SUModelAddMaterials',model,2,(Ref*2)(*materials));geom=self.ref('SUGeometryInputCreate')
        try:
            p=np.array(item['vertices_world_CAD_m'],float)/.0254;points=(Point*len(p))(*(Point(*q) for q in p));self.call('SUGeometryInputSetVertices',geom,len(p),points)
            for i,indices in enumerate(item['triangles']):
                loop=self.ref('SULoopInputCreate')
                try:
                    for index in indices:self.call('SULoopInputAddVertexIndex',loop,int(index))
                    index=Size();self.call('SUGeometryInputAddFace',geom,ct.byref(loop),ct.byref(index))
                finally:
                    if loop.ptr:self.call('SULoopInputRelease',ct.byref(loop))
                front=MaterialInput();front.material=materials[0 if item['top_triangle_mask'][i] else 1];back=MaterialInput();back.material=materials[1]
                self.call('SUGeometryInputFaceSetFrontMaterial',geom,index.value,ct.byref(front));self.call('SUGeometryInputFaceSetBackMaterial',geom,index.value,ct.byref(back))
            self.call('SUEntitiesFill',entities,geom,True)
        finally:self.call('SUGeometryInputRelease',ct.byref(geom))
        self.finish_solid_edges(entities,item)
        return group

    def finish_solid_edges(self,entities,item):
        """Keep physical creases and hide tessellation inside new solids only."""
        p=np.asarray(item['vertices_world_CAD_m'],float);faces=np.asarray(item['triangles'],int)
        q=p[faces];normals=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);normals/=np.linalg.norm(normals,axis=1)[:,None]
        def key(a,b):return tuple(sorted((tuple(np.round(a,6)),tuple(np.round(b,6)))))
        adjacency={}
        for i,tri in enumerate(faces):
            for a,b in zip(tri,np.roll(tri,-1)):adjacency.setdefault(key(p[a],p[b]),[]).append(i)
        for edge in self.list(entities,'Edges'):
            endpoints=[]
            for end in ['Start','End']:
                vertex=self.ref('SUEdgeGet'+end+'Vertex',edge);point=Point();self.call('SUVertexGetPosition',vertex,ct.byref(point));endpoints.append(np.array([point.x,point.y,point.z])*.0254)
            incident=adjacency.get(key(*endpoints),[])
            if len(incident)!=2:continue
            dot=float(normals[incident[0]]@normals[incident[1]])
            # Smooth neighbouring facets on curved surfaces; preserve rims,
            # corners and every open boundary. Coplanar diagonals are hidden.
            if dot>=np.cos(np.deg2rad(15)):
                self.call('SUEdgeSetSoft',edge,True);self.call('SUEdgeSetSmooth',edge,True)
                if dot>1-1e-8:self.hidden('Edge',edge,True)

    def completed_view(self,model,scope=None):
        data=self.geometry(model,scope=scope);p=data['vertices_world_m']
        low,high=p.min(0),p.max(0);target=(low+high)/2;extent=float(np.max(high-low));eye=target+extent*np.array([-.95,-1.,1.15])
        camera=self.ref('SUCameraCreate');e=Point(*(eye/.0254));t=Point(*(target/.0254));up=Point(0,0,1)
        self.call('SUCameraSetOrientation',camera,ct.byref(e),ct.byref(t),ct.byref(up));self.call('SUCameraSetPerspective',camera,True);self.call('SUCameraSetPerspectiveFrustumFOV',camera,45.)
        self.call('SUModelSetCamera',model,ct.byref(camera));scene=self.ref('SUSceneCreate')
        count=self.number('SUModelGetNumScenes',model);names=[]
        if count:
            array=(Ref*count)();n=Size();self.call('SUModelGetScenes',model,count,array,ct.byref(n));names=[self.text('SUSceneGetName',s) for s in list(array)[:n.value]]
        name='Zip2CAD Completed Preview';index=2
        while name in names:name='Zip2CAD Completed Preview '+str(index);index+=1
        self.call('SUSceneSetName',scene,name.encode())
        for name in ['SUSceneSetUseCamera','SUSceneSetUseHiddenGeometry','SUSceneSetUseHiddenObjects']:self.call(name,scene,True)
        self.call('SUModelAddScenes',model,1,(Ref*1)(scene));camera=self.ref('SUModelGetCamera',model);self.call('SUSceneSetCamera',scene,camera);self.call('SUModelSetActiveScene',model,scene)
