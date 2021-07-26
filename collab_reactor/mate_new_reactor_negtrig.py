from numpy.core.fromnumeric import shape
import paramak
from paramak.parametric_components.coolant_channel_ring_curved import CoolantChannelRingCurved
from paramak.utils import union_solid
import paramak_tfcoil_rectroundcorner as ptfc
import numpy as np
from typing import Union, Optional

class NegativeTriangularityReactor(paramak.Reactor):

    """
    This is a new reactor class with negative triangularity.
    """

    def __init__(self,
                inner_tfcoil_thickness: float,
                vacuum_vessel_thickness: float,
                inner_shield_thickness: float,
                plasma_to_wall_gap: float,
                plasma_radial_thickness: float,
                blanket_first_wall_thickness: float,
                breeder_blanket_thickness: float,
                blanket_rear_wall_thickness: float,
                rear_wall_to_inner_pf_gap:float,
                inner_pf_casing_thickness: float,
                inner_pf_to_vacuum_vessel_gap: float,
                outter_tf_coil_thickness: float,
                divertor_radial_thickness: float,
                divertor_height: float,                
                
                outer_blanket_height: float,
                minor_radius: float,
                major_radius: float,
                elongations: float,
                triangularity: float,
                rotation_angle: float,
                divertor_start_angle: float,
                divertor_end_angle: float,
                number_of_coils: int,
                distance: float,
                pf_coil_heights: Union[float,list],
                pf_coil_widths: Union[float,list],
                pf_coil_center_points: Union[list,tuple],

                rear_blanket_thickness: Optional[float] = None,

    ) -> None:
        
        super().__init__([])
        self._inner_tfcoil_thickness = inner_tfcoil_thickness
        self._vacuum_vessel_thickness = vacuum_vessel_thickness
        self._inner_shield_thickness = inner_shield_thickness
        self._plasma_to_wall_gap = plasma_to_wall_gap
        self._plasma_radial_thickness = plasma_radial_thickness
        self._divertor_radial_thickness = divertor_radial_thickness
        self._divertor_height = divertor_height
        self._blanket_first_wall_thickness = blanket_first_wall_thickness
        self._breeder_blanket_thickness = breeder_blanket_thickness
        self._blanket_rear_wall_thickness = blanket_rear_wall_thickness
        self._rear_blanket_thickness = rear_blanket_thickness
        self._elongation = elongations
        self._triangularity = triangularity
        self._rotation_angle = rotation_angle
        self._outer_blanket_height = outer_blanket_height
        self._divertor_start_angle = divertor_start_angle
        self._divertor_end_angle = divertor_end_angle
        self._number_of_coils = number_of_coils
        self._distance = distance
        self._pf_coil_heights = pf_coil_heights
        self._pf_coil_widths = pf_coil_widths
        self._pf_coil_center_points = pf_coil_center_points

        self._minor_radius = minor_radius
        self._major_radius = major_radius

    def create_solids(self):

        shapes_and_components = []

        self._make_vertical_build()
        self._make_radial_build()
        self._make_plasma() # make it optional to be added to reactor geometry

        shapes_and_components.append(self._make_inner_tfcoil_leg())
        shapes_and_components.append(self._make_vaccuum_vessel_inner_wall())
        shapes_and_components.append(self._make_inner_shield())

        shapes_and_components += self._make_blankets()
        shapes_and_components += self._make_divertor()

        shapes_and_components.append(self._make_plasma())
        shapes_and_components.append(self._make_vacuum_vessel())

        shapes_and_components.append(self._make_tfcoils())
        shapes_and_components.append(self._make_pf_coils())
        
        self.shapes_and_components = shapes_and_components

    def _make_vertical_build(self):

        self._inner_tf_leg_height = ((self._minor_radius * self._elongation) \
                                + self._plasma_to_wall_gap \
                                + self._blanket_first_wall_thickness \
                                + self._breeder_blanket_thickness \
                                + self._blanket_rear_wall_thickness \
                                + self._divertor_height \
                                + self._vacuum_vessel_thickness) * 2

        self._vaccuum_vessel_inner_wall_height = ((self._minor_radius * self._elongation) \
                                + self._plasma_to_wall_gap \
                                + self._blanket_first_wall_thickness \
                                + self._breeder_blanket_thickness \
                                + self._blanket_rear_wall_thickness \
                                + self._divertor_height) * 2

    def _make_radial_build(self):
        
        self._vacuum_vessel_inner_wall_start_rad = self._inner_tfcoil_thickness
        self._vacuum_vessel_inner_wall_end_rad = self._vacuum_vessel_inner_wall_start_rad + self._vacuum_vessel_thickness

        self._inner_shield_start_rad = self._vacuum_vessel_inner_wall_end_rad
        self._inner_shield_end_rad = self._inner_shield_start_rad + self._inner_shield_thickness

        self._divertor_radial_end = self._major_radius + self._minor_radius \
                                - self._blanket_first_wall_thickness \
                                - self._breeder_blanket_thickness \
                                - self._blanket_rear_wall_thickness \
                                - self._plasma_to_wall_gap
        self._divertor_radial_start = self._divertor_radial_end - self._divertor_radial_thickness




    def _make_inner_tfcoil_leg(self):

        tf_inner_leg = paramak.CenterColumnShieldCylinder(
            height=self._inner_tf_leg_height,
            inner_radius=0,
            outer_radius=self._inner_tfcoil_thickness,
            rotation_angle=self._rotation_angle,
            stp_filename="tf_inner_leg.stp",
            stl_filename="tf_inner_leg.stl",
            name="tf_inner_leg",
            material_tag="tf_coil_mat",
            color=(0.2,1,0.2),
        )
        return tf_inner_leg

    def _make_vaccuum_vessel_inner_wall(self):

        inner_vacuum_wall = paramak.CenterColumnShieldCylinder(
            height=self._vaccuum_vessel_inner_wall_height,
            inner_radius=self._vacuum_vessel_inner_wall_start_rad,
            outer_radius=self._vacuum_vessel_inner_wall_end_rad,
            rotation_angle=self._rotation_angle,
            stp_filename="vacuum_vessel_inner_wall.stp",
            stl_filename="vacuum_vessel_inner_wall.stl",
            name="vacuum_vessel_inner_wall",
            material_tag="vacuum_vessel_mat",
            color=(0.5,0.5,0.5)
        )
        return inner_vacuum_wall

    def _make_inner_shield(self):

        inner_shield = paramak.CenterColumnShieldCylinder(
            height=self._vaccuum_vessel_inner_wall_height,
            inner_radius=self._inner_shield_start_rad,
            outer_radius=self._inner_shield_end_rad,
            rotation_angle=self._rotation_angle,
            stp_filename="inner_shield.stp",
            stl_filename="inner_shield.stl",
            name="inner_shield",
            material_tag="vacuum_vessel_mat",
            color=(1,0.7,0.5)
        )
        return inner_shield
    
    def _make_plasma(self):

        plasma = paramak.Plasma(
            major_radius=self._major_radius,
            minor_radius=self._minor_radius,
            elongation=self._elongation,
            triangularity=self._triangularity,
            rotation_angle=self._rotation_angle,
            color=(0,0.5,0.5)
        )
        self._plasma = plasma
        return plasma


    def _make_blankets(self):

        self._center_cutter = paramak.CenterColumnShieldCylinder(
            height=self._vaccuum_vessel_inner_wall_height,
            inner_radius=0,
            outer_radius=self._inner_shield_end_rad,
            rotation_angle=self._rotation_angle,
            color=(0,0,0)
        )

        self._first_outer_blanket_wall = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_first_wall_thickness,
            offset_from_plasma=self._plasma_to_wall_gap,
            start_angle=self._divertor_start_angle ,
            stop_angle=-self._divertor_start_angle,
            rotation_angle=self._rotation_angle,
            material_tag="firstwall_mat",
            stp_filename="firstwall.stp",
            stl_filename="firstwall.stl",
            name="firstwall",
            cut=[self._center_cutter],
            color=(0.7,0.7,0.7)

        )

        self._first_outer_blanket = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._breeder_blanket_thickness,
            offset_from_plasma=self._plasma_to_wall_gap + self._blanket_first_wall_thickness,
            start_angle=self._divertor_start_angle,
            stop_angle=-self._divertor_start_angle,
            rotation_angle=self._rotation_angle,
            material_tag="blanket_mat",
            stp_filename="blanket.stp",
            stl_filename="blanket.stl",
            name="blanket",
            cut=[self._center_cutter],
            color=(0,1,0.45)

        )

        self._first_outer_blanket_rearwall = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_rear_wall_thickness,
            offset_from_plasma=self._plasma_to_wall_gap + self._blanket_first_wall_thickness + self._breeder_blanket_thickness,
            start_angle=self._divertor_start_angle,
            stop_angle=-self._divertor_start_angle,
            rotation_angle=self._rotation_angle,
            material_tag="blanket_mat",
            stp_filename="blanket.stp",
            stl_filename="blanket.stl",
            name="blanket",
            cut=[self._center_cutter],
            color=(0.75,0.75,0.75)

        )

        self._first_inner_blanket_wall = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_first_wall_thickness,
            offset_from_plasma=self._plasma_to_wall_gap,
            start_angle=self._divertor_end_angle,
            stop_angle=360 - self._divertor_end_angle,
            rotation_angle=self._rotation_angle,
            material_tag="firstwall_mat",
            stp_filename="firstwall.stp",
            stl_filename="firstwall.stl",
            name="firstwall",
            cut=[self._center_cutter],
            color=(0.7,0.7,0.7)

        )

        self._first_inner_blanket = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._breeder_blanket_thickness,
            offset_from_plasma=self._plasma_to_wall_gap + self._blanket_first_wall_thickness,
            start_angle=self._divertor_end_angle,
            stop_angle=360 - self._divertor_end_angle,
            rotation_angle=self._rotation_angle,
            material_tag="blanket_mat",
            stp_filename="blanket.stp",
            stl_filename="blanket.stl",
            name="blanket",
            cut=[self._center_cutter],
            color=(0,1,0.45)

        )

        self._first_inner_blanket_rearwall = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_rear_wall_thickness,
            offset_from_plasma=self._plasma_to_wall_gap + self._blanket_first_wall_thickness + self._breeder_blanket_thickness,
            start_angle=self._divertor_end_angle,
            stop_angle=360 - self._divertor_end_angle,
            rotation_angle=self._rotation_angle,
            material_tag="blanket_mat",
            stp_filename="blanket.stp",
            stl_filename="blanket.stl",
            name="blanket",
            cut=[self._center_cutter],
            color=(0.75,0.75,0.75)

        )


        return [self._first_outer_blanket_wall, self._first_outer_blanket,self._first_outer_blanket_rearwall,self._first_inner_blanket_wall,self._first_inner_blanket_rearwall,self._first_inner_blanket]


    def _make_divertor(self):

        self._shell = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_first_wall_thickness + self._blanket_rear_wall_thickness + self._breeder_blanket_thickness,
            offset_from_plasma=self._plasma_to_wall_gap,
            start_angle=self._divertor_start_angle,
            stop_angle=-self._divertor_start_angle,
            rotation_angle=self._rotation_angle,
            color=(1,0,0)
        )

        self._inside = paramak.Plasma(
            major_radius=self._major_radius,
            minor_radius=self._minor_radius + self._plasma_to_wall_gap,
            elongation=self._elongation,
            triangularity=self._triangularity,
            rotation_angle=self._rotation_angle,
            union=[self._shell],
            color=(0,0.5,0.5)
        )

        self._divertor_ext_top = paramak.PoloidalFieldCoil(
            height=self._divertor_height*2,
            width=self._divertor_radial_thickness,
            center_point=(self._plasma.high_point[0] - self._plasma_to_wall_gap,
                    self._plasma.high_point[1] + self._plasma_to_wall_gap + self._blanket_first_wall_thickness + self._blanket_rear_wall_thickness + self._breeder_blanket_thickness),
            rotation_angle=self._rotation_angle,
            cut=[self._inside]
        )

        self._divertor_ext_bot = paramak.PoloidalFieldCoil(
            height=self._divertor_height*2,
            width=self._divertor_radial_thickness,
            center_point=(self._plasma.low_point[0] - self._plasma_to_wall_gap,
                    self._plasma.low_point[1] - (self._plasma_to_wall_gap + self._blanket_first_wall_thickness + self._blanket_rear_wall_thickness + self._breeder_blanket_thickness)),
            rotation_angle=self._rotation_angle,
            cut=[self._inside]
        )


        self._divertor_top = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_first_wall_thickness + self._blanket_rear_wall_thickness + self._breeder_blanket_thickness,
            offset_from_plasma=self._plasma_to_wall_gap,
            start_angle=self._divertor_start_angle,
            stop_angle=self._divertor_end_angle,
            rotation_angle=self._rotation_angle,
            material_tag="divertor_mat",
            stp_filename="divertor.stp",
            stl_filename="divertor.stl",
            name="divertor",
            cut=[self._center_cutter],
            union=[self._divertor_ext_top],
            color=(1,0,0)
        )
        

        self._divertor_bot = paramak.BlanketFP(
            plasma=self._plasma,
            thickness= self._blanket_first_wall_thickness + self._blanket_rear_wall_thickness + self._breeder_blanket_thickness,
            offset_from_plasma=self._plasma_to_wall_gap,
            start_angle=-self._divertor_start_angle,
            stop_angle=-self._divertor_end_angle,
            rotation_angle=self._rotation_angle,
            material_tag="divertor_mat",
            stp_filename="divertor.stp",
            stl_filename="divertor.stl",
            name="divertor",
            cut=[self._center_cutter],
            union=[self._divertor_ext_bot],
            color=(1,0,0)
        )

        return [self._divertor_top, self._divertor_bot]


    def _make_vacuum_vessel(self):

        self._most_outer_point_of_blanket = self._plasma.outer_equatorial_point[0] \
            + self._plasma_to_wall_gap \
            + self._blanket_first_wall_thickness \
            + self._breeder_blanket_thickness \
            + self._blanket_rear_wall_thickness

        self._vac_ves_cut = paramak.CenterColumnShieldCylinder(
            height=self._vaccuum_vessel_inner_wall_height,
            inner_radius=0,
            outer_radius=self._most_outer_point_of_blanket,
            rotation_angle=self._rotation_angle,
            color=(0,0,0)
        )

        self._vac_ves_cut2 = self._vac_ves_cut = paramak.CenterColumnShieldCylinder(
            height=self._inner_tf_leg_height,
            inner_radius=0,
            outer_radius=self._inner_tfcoil_thickness,
            rotation_angle=self._rotation_angle,
            union=[self._vac_ves_cut],
            color=(0,0,0)
        )

        self._vac_ves = paramak.CenterColumnShieldCylinder(
            height=self._vaccuum_vessel_inner_wall_height + self._vacuum_vessel_thickness*2,
            inner_radius=0,
            outer_radius=self._most_outer_point_of_blanket + self._vacuum_vessel_thickness,
            rotation_angle=self._rotation_angle,
            material_tag="vacuum_vessel_mat",
            stp_filename="vacuum_vessel.stp",
            stl_filename="vacuum_vessel.stl",
            name="vacuum_vessel",
            cut=[self._vac_ves_cut2],
            color=(0.5,0.5,0.5)
        )

        return self._vac_ves



    def _make_tfcoils(self):

        self._tf_coils = ptfc.ToroidalFieldCoilRectangleRoundCorners(
            with_inner_leg=False,
            lower_inner_coordinates=(0, -self._inner_tf_leg_height/2),
            mid_point_coordinates=(self._most_outer_point_of_blanket + self._vacuum_vessel_thickness,0 ),
            thickness=self._inner_tfcoil_thickness,
            number_of_coils=self._number_of_coils,
            distance=self._distance,
            stp_filename="tf_coil.stp",
            name="tf_coil",
            material_tag="tf_coil_mat",
            stl_filename="tf_coil.stl",
            rotation_angle=self._rotation_angle
        )

        return self._tf_coils


    def _make_pf_coils(self):

        self._pf_coils = paramak.PoloidalFieldCoilSet(
            heights=self._pf_coil_heights,
            widths=self._pf_coil_widths,
            center_points=self._pf_coil_center_points,
            stp_filename="pf_coil_set.stp",
            stl_filename="pf_coil_set.stl",
            name="pf_coil_set",
            color=(0.7,0.7,0.2),
            rotation_angle=self._rotation_angle

        )
        return self._pf_coils



#if __name__ == "__main__":    

obj = NegativeTriangularityReactor(
    inner_tfcoil_thickness = 50,
    vacuum_vessel_thickness = 20,
    inner_shield_thickness = 10,
    plasma_to_wall_gap = 10,
    plasma_radial_thickness = 60,
    divertor_radial_thickness = 250,
    divertor_height = 100,
    blanket_first_wall_thickness = 20,
    breeder_blanket_thickness = 50,
    blanket_rear_wall_thickness = 20,
    outer_blanket_height=100,
    minor_radius = 150,
    major_radius = 250,
    elongations = 2,
    triangularity = -0.55,
    rotation_angle = 180,
    divertor_start_angle=65,
    divertor_end_angle=125,
    number_of_coils = 12,
    distance = 50,
    pf_coil_heights = [60, 60, 60, 60, 60],
    pf_coil_widths = [60, 60, 60, 60, 60],
    pf_coil_center_points = [(600,500), (600,250), (600,0), (600,-250), (600,-500)],
        
)

obj.create_solids()
obj.show()
