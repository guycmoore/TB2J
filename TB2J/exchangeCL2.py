"""
Exchange from Green's function
"""

import os
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from TB2J.external import p_map
from TB2J.green import TBGreen
from TB2J.io_exchange import SpinIO

from .exchange import ExchangeCL


class ExchangeCL2(ExchangeCL):
    def set_tbmodels(self, tbmodels):
        """
        only difference is a colinear tag.
        """
        self.tbmodel_up, self.tbmodel_dn = tbmodels
        self.backend_name = self.tbmodel_up.name
        self.Gup = TBGreen(
            tbmodel=self.tbmodel_up,
            kmesh=self.kmesh,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
        )
        self.Gdn = TBGreen(
            tbmodel=self.tbmodel_dn,
            kmesh=self.kmesh,
            efermi=self.efermi,
            use_cache=self._use_cache,
            nproc=self.nproc,
        )
        if self.write_density_matrix:
            self.Gup.write_rho_R(
                Rlist=self.Rlist, fname=os.path.join(self.output_path, "rho_up.pickle")
            )
            self.Gdn.write_rho_R(
                Rlist=self.Rlist, fname=os.path.join(self.output_path, "rho_dn.pickle")
            )
        self.norb = self.Gup.norb
        self.nbasis = self.Gup.nbasis + self.Gdn.nbasis
        # self.rho_up_list = []
        # self.rho_dn_list = []
        self.rho_up = self.Gup.get_density_matrix()
        self.rho_dn = self.Gdn.get_density_matrix()
        self.Jorb_full_list = defaultdict(lambda: defaultdict(lambda: []))
        self.Jorb_list = defaultdict(lambda: [])
        self.JJ_list = defaultdict(lambda: [])
        self.JJ = defaultdict(lambda: 0.0j)
        self.Jorb_full = defaultdict(lambda: defaultdict(lambda: 0.0j))
        self.Jorb = defaultdict(lambda: 0.0j)
        self.HR0_up = self.Gup.H0
        self.HR0_dn = self.Gdn.H0
        self.Delta = self.HR0_up - self.HR0_dn
        if self.Gup.is_orthogonal and self.Gdn.is_orthogonal:
            self.is_orthogonal = True
        else:
            self.is_orthogonal = False
            # self.S0=self.Gup.S0

        self.exchange_Jdict = {}
        self.Jiso_orb = {}
        self.Jiso_orb_full = {}

        self.biquadratic = False

        self._is_collinear = True

    def _clean_tbmodels(self):
        del self.tbmodel_up
        del self.tbmodel_dn
        del self.Gup.tbmodel
        del self.Gdn.tbmodel

    def _adjust_emin(self):
        emin_up = self.Gup.find_energy_ingap(rbound=self.efermi - 5.0) - self.efermi
        emin_dn = self.Gdn.find_energy_ingap(rbound=self.efermi - 5.0) - self.efermi
        self.emin = min(emin_up, emin_dn)
        print(f"A gap is found at {self.emin}, set emin to it.")

    def get_Delta(self, iatom):
        orbs = self.iorb(iatom)
        return self.Delta[np.ix_(orbs, orbs)]
        # s = self.orb_slice[iatom]
        # return self.Delta[s, s]

    def GR_atom(self, GR, iatom, jatom):
        """Given a green's function matrix, return the [iatom, jatom] component.

        :param GR:  Green's function matrix
        :param iatom:  index of atom i
        :param jatom:  index of atom j
        :returns:  G_ij
        :rtype:  complex matrix.
        """
        orbi = self.iorb(iatom)
        orbj = self.iorb(jatom)
        return GR[np.ix_(orbi, orbj)]
        # return GR[self.orb_slice[iatom], self.orb_slice[jatom]]

    def get_A_ijR(self, Gup, Gdn, iatom, jatom):
        """
        ! Note: not used. In get_all_A, it is reimplemented.
        """
        Rij_done = set()
        Jorb_list = dict()
        JJ_list = dict()
        for R, ijpairs in self.R_ijatom_dict.items():
            if (iatom, jatom) in ijpairs and (R, iatom, jatom) not in Rij_done:
                Gij_up = self.GR_atom(Gup[R], iatom, jatom)
                Rm = tuple(-x for x in R)
                Gji_dn = self.GR_atom(Gdn[Rm], jatom, iatom)
                tmp = 0.0j
                Deltai = self.get_Delta(iatom)
                Deltaj = self.get_Delta(jatom)
                t = np.einsum(
                    "ij, ji-> ij", np.matmul(Deltai, Gij_up), np.matmul(Deltaj, Gji_dn)
                )

                # if self.biquadratic:
                #    A = np.einsum(
                #        "ij, ji-> ij",
                #        np.matmul(Deltai, Gij_up),
                #        np.matmul(Deltaj, Gji_up),
                #    )
                #    C = np.einsum(
                #        "ij, ji-> ij",
                #        np.matmul(Deltai, Gij_down),
                #        np.matmul(Deltaj, Gji_down),
                #    )
                tmp = np.sum(t)
                self.Jorb_list[(R, iatom, jatom)].append(t / (4.0 * np.pi))
                self.JJ_list[(R, iatom, jatom)].append(tmp / (4.0 * np.pi))
                Rij_done.add((R, iatom, jatom))
                if (Rm, jatom, iatom) not in Rij_done:
                    Jorb_list[(Rm, jatom, iatom)] = t.T / (4.0 * np.pi)
                    JJ_list[(Rm, jatom, iatom)] = tmp / (4.0 * np.pi)
                    Rij_done.add((Rm, jatom, iatom))
        return Jorb_list, JJ_list

    def get_all_A(self, Gup, Gdn):
        """
        Calculate all A matrix elements
        Loop over all magnetic atoms.
        :param G: Green's function.
        :param de: energy step.
        """
        Rij_done = set()
        Jorb_full_list = dict()
        Jorb_list = dict()
        JJ_list = dict()
        for R, ijpairs in self.R_ijatom_dict.items():
            for iatom, jatom in ijpairs:
                if (R, iatom, jatom) not in Rij_done:
                    Rm = tuple(-x for x in R)
                    if (Rm, jatom, iatom) in Rij_done:
                        raise KeyError(
                            f"Strange (Rm, jatom, iatom) has already been calculated! {(Rm, jatom, iatom)}"
                        )
                    
                    Gij_up = self.GR_atom(Gup[R], iatom, jatom)
                    Gji_dn = self.GR_atom(Gdn[Rm], jatom, iatom)
                    
                    Del_i, Del_j = self.get_Delta(iatom), self.get_Delta(jatom)
                    
                    d_tmp, d_tmp_t = dict(), dict()
                    tmp_exch = 0.0j
                    for m0 in range(Del_i.shape[0]):
                        for m1 in range(Del_j.shape[0]):
                            for m2 in range(Del_i.shape[1]):
                                for m3 in range(Del_j.shape[1]):
                                    
                                    tmp_exch = Del_i[m0,m1] * Gij_up[m1,m2] * Del_j[m2,m3] * Gji_dn[m3,m0]
                                    # tmp_exch = Del_i[m0,m1] * Gij_up[m1,m2] * Del_j[m3,m2] * Gji_dn[m0,m3]
                                    
                                    tmp_exch = tmp_exch / (4.0 * np.pi)
                                    d_tmp[(m0, m1, m2, m3)] = tmp_exch
                                    d_tmp_t[(m2, m3, m0, m1)] = tmp_exch
                    
                    tmp = 0.0j
                    # t = self.get_Delta(iatom) @ Gij_up @ self.get_Delta(jatom) @ Gji_dn
                    t = np.einsum(
                        "ij, ji-> ij",
                        np.matmul(self.get_Delta(iatom), Gij_up),
                        np.matmul(self.get_Delta(jatom), Gji_dn),
                    )
                    tmp = np.sum(t)
                    
                    Jorb_full_list[(R, iatom, jatom)] = d_tmp
                    Jorb_list[(R, iatom, jatom)] = t / (4.0 * np.pi)
                    JJ_list[(R, iatom, jatom)] = tmp / (4.0 * np.pi)
                    Rij_done.add((R, iatom, jatom))
                    if (Rm, jatom, iatom) not in Rij_done:
                        Jorb_full_list[(Rm, jatom, iatom)] = d_tmp_t
                        Jorb_list[(Rm, jatom, iatom)] = t.T / (4.0 * np.pi)
                        JJ_list[(Rm, jatom, iatom)] = tmp / (4.0 * np.pi)
                        Rij_done.add((Rm, jatom, iatom))
        return Jorb_full_list, Jorb_list, JJ_list

    def A_to_Jtensor(self):
        for key, val in self.JJ.items():
            # key:(R, iatom, jatom)
            R, iatom, jatom = key
            ispin = self.ispin(iatom)
            jspin = self.ispin(jatom)
            keyspin = (R, ispin, jspin)
            is_nonself = not (R == (0, 0, 0) and iatom == jatom)
            Jij = np.imag(val) / np.sign(np.dot(self.spinat[iatom], self.spinat[jatom]))
            Jorbij = np.imag(self.Jorb[key]) / np.sign(
                np.dot(self.spinat[iatom], self.spinat[jatom])
            )
            Jorbij_full = dict()
            for jokey, joval in self.Jorb_full[key].items():
                Jorbij_full[jokey] = np.imag(joval)
            if True:
            # if is_nonself:
                self.exchange_Jdict[keyspin] = Jij
                self.Jiso_orb[keyspin] = self.simplify_orbital_contributions(
                    Jorbij, iatom, jatom
                )
                self.Jiso_orb_full[keyspin] = Jorbij_full.copy()

    def get_rho_e(self, rho_up, rho_dn):
        # self.rho_up_list.append(-1.0 / np.pi * np.imag(rho_up[(0,0,0)]))
        # self.rho_dn_list.append(-1.0 / np.pi * np.imag(rho_dn[(0,0,0)]))
        rup = -1.0 / np.pi * rho_up[(0, 0, 0)]
        rdn = -1.0 / np.pi * rho_dn[(0, 0, 0)]
        return rup, rdn

    def get_rho_atom(self):
        """
        charges and spins from density matrices
        """
        self.charges = np.zeros(len(self.atoms), dtype=float)
        self.spinat = np.zeros((len(self.atoms), 3), dtype=float)
        for iatom in self.orb_dict:
            iorb = self.iorb(iatom)
            tup = np.real(np.trace(self.rho_up[np.ix_(iorb, iorb)]))
            tdn = np.real(np.trace(self.rho_dn[np.ix_(iorb, iorb)]))
            self.charges[iatom] = tup + tdn
            self.spinat[iatom, 2] = tup - tdn

    def finalize(self):
        self.Gup.clean_cache()
        self.Gdn.clean_cache()
        # path = 'TB2J_results/cache'
        # if os.path.exists(path):
        #    shutil.rmtree(path)

    def integrate(self, method="simpson"):
        # if method == "trapezoidal":
        #    integrate = trapezoidal_nonuniform
        # elif method == "simpson":
        #    integrate = simpson_nonuniform
        for R, ijpairs in self.R_ijatom_dict.items():
            for iatom, jatom in ijpairs:
                # self.Jorb[(R, iatom, jatom)] = integrate(
                #    self.contour.path, self.Jorb_list[(R, iatom, jatom)]
                # )
                # self.JJ[(R, iatom, jatom)] = integrate(
                #    self.contour.path, self.JJ_list[(R, iatom, jatom)]
                # )
                for jokey, joval in self.Jorb_full_list[(R, iatom, jatom)].items():
                    self.Jorb_full[(R, iatom, jatom)][jokey] = self.contour.integrate_values(
                        joval
                    )
                self.Jorb[(R, iatom, jatom)] = self.contour.integrate_values(
                    self.Jorb_list[(R, iatom, jatom)]
                )
                self.JJ[(R, iatom, jatom)] = self.contour.integrate_values(
                    self.JJ_list[(R, iatom, jatom)]
                )

    def get_quantities_per_e(self, e):
        GR_up = self.Gup.get_GR(self.short_Rlist, energy=e, get_rho=False)
        GR_dn = self.Gdn.get_GR(self.short_Rlist, energy=e, get_rho=False)
        Jorb_full_list, Jorb_list, JJ_list = self.get_all_A(GR_up, GR_dn)
        return dict(Jorb_full_list=Jorb_full_list, Jorb_list=Jorb_list, JJ_list=JJ_list)

    def calculate_all(self):
        """
        The top level.
        """
        print("Green's function Calculation started.")

        npole = len(self.contour.path)
        if self.nproc == 1:
            results = map(
                self.get_quantities_per_e, tqdm(self.contour.path, total=npole)
            )
        else:
            # pool = ProcessPool(nodes=self.nproc)
            # results = pool.map(self.get_AijR_rhoR ,self.contour.path)
            results = p_map(
                self.get_quantities_per_e, self.contour.path, num_cpus=self.nproc
            )
        for i, result in enumerate(results):
            Jorb_full_list = result["Jorb_full_list"]
            Jorb_list = result["Jorb_list"]
            JJ_list = result["JJ_list"]
            for iR, R in enumerate(self.R_ijatom_dict):
                for iatom, jatom in self.R_ijatom_dict[R]:
                    key = (R, iatom, jatom)
                    for jokey,joval in Jorb_full_list[key].items():
                        self.Jorb_full_list[key][jokey].append(joval)
                    self.Jorb_list[key].append(Jorb_list[key])
                    self.JJ_list[key].append(JJ_list[key])
        self.integrate()
        self.get_rho_atom()
        self.A_to_Jtensor()

    def write_output(self, path="TB2J_results"):
        self._prepare_index_spin()
        output = SpinIO(
            atoms=self.atoms,
            charges=self.charges,
            spinat=self.spinat,
            index_spin=self.index_spin,
            colinear=True,
            orbital_names=self.orbital_names,
            distance_dict=self.distance_dict,
            exchange_Jdict=self.exchange_Jdict,
            Jiso_orb=self.Jiso_orb,
            Jiso_orb_full=self.Jiso_orb_full,
            dmi_ddict=None,
            NJT_Jdict=None,
            NJT_ddict=None,
            Jani_dict=None,
            biquadratic_Jdict=None,
            description=self.description,
        )
        output.write_all(path=path)
