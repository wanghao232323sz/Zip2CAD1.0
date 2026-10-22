"""Train-ranked duplicate suppression for schematic structural proposals."""
import numpy as np,cv2

def suppress_duplicate_components(rows,parts,plane_tolerance):
    selected=[];suppressed=[]
    order=sorted((r for r in rows if r['native_edit_approved']),key=lambda r:(-r['training_voxel_point_count'],-r['occupied_area_m2'],r['component_id']))
    for row in order:
        poly=np.asarray(row['boundary_xy'],np.float32);area=cv2.contourArea(poly);duplicate=None
        for prior in selected:
            q=np.asarray(prior['boundary_xy'],np.float32);low=np.minimum(poly.min(0),q.min(0));high=np.maximum(poly.max(0),q.max(0));step=.25
            yy,xx=np.mgrid[low[1]+step/2:high[1]:step,low[0]+step/2:high[0]:step];points=np.c_[xx.ravel(),yy.ravel()]
            a=np.array([cv2.pointPolygonTest(poly,tuple(map(float,p)),False)>=0 for p in points]);b=np.array([cv2.pointPolygonTest(q,tuple(map(float,p)),False)>=0 for p in points])
            overlap=float((a&b).sum()/max(1,min(a.sum(),b.sum())))
            n=np.asarray(row['plane_normal']);pn=np.asarray(prior['plane_normal']);gap=abs((np.asarray(row['plane_origin'])-np.asarray(prior['plane_origin']))@pn)
            if overlap>=.7 and abs(n@pn)>=.96 and gap<=2*plane_tolerance:
                duplicate={'component_id':row['component_id'],'retained_component_id':prior['component_id'],'smaller_area_overlap_fraction':overlap,
                    'plane_separation_m':float(gap),'reason':'duplicate_training_supported_surface_hypothesis','heldout_used_for_ranking':False};break
        if duplicate:
            row['native_edit_approved']=False;row['duplicate_suppression']=duplicate;suppressed.append(duplicate)
        else:selected.append(row)
    names={r['component_id'] for r in selected};return [p for p in parts if p['component_id'] in names],suppressed
